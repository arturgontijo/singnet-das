import logging
import amrlib
import penman
import penman.models.noop
import re

from amrlib.alignments.rbw_aligner import RBWAligner
from amrlib.graph_processing.annotator import add_lemmas, load_spacy

from amratom.atomese import is_variable, is_amrset_name

_spacy_model = 'en_core_web_md'
_penman_model = penman.models.noop.NoOpModel()

def _load_spacy():
    #return spacy.load(_spacy_model)
    # There is no API to inject own model into amrlib pipeline. I keep this
    # code because I don't like reimplementing internal `add_lemmas` logic
    # using own spacy model
    load_spacy(_spacy_model)
    from amrlib.graph_processing.annotator import spacy_nlp
    logging.getLogger(__name__).debug("_load_spacy(): %s is loaded",
            spacy_nlp.path)
    return spacy_nlp

_stog_model_cache = None
_gtos_model_cache = None
_spacy_model_cache = None

def load_models():
    global _stog_model_cache
    if _stog_model_cache is None:
        _stog_model_cache = amrlib.load_stog_model()

    global _gtos_model_cache
    if _gtos_model_cache is None:
        _gtos_model_cache = amrlib.load_gtos_model()

    global _spacy_model_cache
    if _spacy_model_cache is None:
        _spacy_model_cache = _load_spacy()

    amrlib.setup_spacy_extension()

class AmrProcessor:

    def __init__(self):
        self.log = logging.getLogger(__name__ + '.' + type(self).__name__)

        load_models()

        global _spacy_model_cache
        self.nlp = _spacy_model_cache
        global _gtos_model_cache
        self.gtos = _gtos_model_cache

    def utterance_to_amr(self, utterance, indent=-1):
        doc = self.nlp(utterance)
        amrs = doc._.to_amr()
        sents = doc.sents
        triples_proc = []
        for p in zip(amrs, sents):
            # Entry point (amr text->triples); can be replaced with penman.decode
            triples, top = self._add_pos_tags(doc, p[0], p[1], indent)
            # further triples processing
            triples_proc += self._sentence_splitter(triples, top)
        return list(map(lambda ts: penman.encode(penman.Graph(ts), indent=indent, model=_penman_model),
                        triples_proc))

    def amr_to_utterance(self, amr):
        return self.gtos.generate([amr], use_tense=False)[0][0]

    def triples_to_utterance(self, triples):
        return self.amr_to_utterance(penman.encode(penman.Graph(triples)))

    def _add_pos_tags(self, doc, amr, sent, indent):
        self.log.debug('_add_pos_tags: amr: %s, sent: %s', amr, sent)
        graph = add_lemmas(amr, snt_key='snt')
        aligner = RBWAligner.from_penman_w_json(graph)
        graph = aligner.get_penman_graph()
        triples = graph.triples
        self.log.debug('_add_pos_tags: alignments: %s',
                penman.surface.alignments(graph))
        for triple, alignment in penman.surface.alignments(graph).items():
            pos_tag = doc[sent.start + alignment.indices[0]].tag_
            triples.append((triple[0], ':pos', '"' + pos_tag + '"'))
        self.log.debug('_add_pos_tags: triples: %s', triples)
        return triples, graph.top

    def _child_lnodes_rec(self, triples, parent):
        '''
        Returns all those nodes (possibly duplicated) in the subgraph (including `parent`)
        which appear on the left side of triples (i.e. not leaves)
        '''
        grandchildren = []
        isOnLeft = False
        for t in triples:
            if t[0] == parent:
                isOnLeft = True
                grandchildren += self._child_lnodes_rec(triples, t[2])
        return [parent] + grandchildren if isOnLeft else []

    def _sentence_splitter(self, triples, top):
        top_roles = []
        top_concept = None
        for triple in triples:
            if triple[0] == top:
                if triple[1] == ':instance':
                    top_concept = triple[2]
                elif triple[1] != ':pos':
                    top_roles += [(triple[1], triple[2])]
        if top_concept == 'and':
            expected = 'op'
        elif top_concept == 'multi-sentence':
            expected = 'snt'
        else: return [triples]
        # Just checking that there are no unexpected roles
        for r in top_roles:
            if not expected in r[0]:
                logging.getLogger(__name__).debug("_sentence_splitter(): WARNING - unexpected role %s for %s",
                                                r[0], top_concept)
                return [triples]
        subgraphs = [[[], self._child_lnodes_rec(triples, r[1])] for r in top_roles]
        for t in triples:
            for s in subgraphs:
                if t[0] in s[1]:
                    s[0] += [t]
        return [s[0] for s in subgraphs]


def is_instance(triple):
    return triple[1] == ':instance'

def is_unknown(triple):
    return is_instance(triple) and triple[2] == 'amr-unknown'

def _remove_postfix(word):
    pos = word.rfind('-')
    return word[:pos] if pos > 0 else word

def triple_to_string(triple):
    source, role, target = triple
    return '(' + source + ', ' + role + ', ' + target + ')'

def triple_from_string(line):
    left = line.find('(')
    right = line.rfind(')')
    source, role, target = line[left+1:right].split(', ')
    return (source, role, target)

class AmrInstanceDict:

    def __init__(self, id_generator):
        self.log = logging.getLogger(__name__ + '.AmrInstanceDict')
        self.id_generator = id_generator
        self.instance_by_node = {}
        self.instance_triples = []

    def add_graph(self, graph):
        for triple in filter(is_instance, graph.triples):
            self.log.debug('triple: %s', triple)
            (source, role, target) = triple
            instance = self._get_unique_instance(target)
            self.instance_by_node[source] = instance
            self.instance_triples.append((instance, role, target))

    def _get_unique_instance(self, target):
        id = self.id_generator()
        return target + '-' + '{:06d}'.format(id)

    def get_instance_triples(self):
        return self.instance_triples

    def map_node_to_instance(self, node):
        if node in self.instance_by_node:
            return self.instance_by_node[node]
        else:
            return node

_number_or_string_pattern = re.compile(r'\d+(\.\d+)?|"[^\"]+"|-|\+')

def is_amr_set(triple):
    return triple[1] == ':amr-set'

def is_const( word):
    return _number_or_string_pattern.fullmatch(word)

_roles_with_attrs_at_right = { ':mode', ':pos', ':polarity' }

class PatternInstanceDict(AmrInstanceDict):

    def __init__(self, id_generator):
        super().__init__(id_generator)
        self.log = logging.getLogger(__name__ + '.PatternInstanceDict')

    def add_graph(self, graph):
        for triple in filter(is_instance, graph.triples):
            node, instance_role, concept = triple
            assert not(is_variable(node) and is_amrset_name(concept)), (
                '($var / @amrset) is not supported')
            assert not(node == '-' and is_variable(concept)), (
                '(- / $var) is not supported')
            if concept is None:
                continue
            if concept == '-':
                self.instance_by_node[node] = node
                continue
            instance = self._get_unique_instance(concept)
            self.instance_by_node[node] = instance
            self.instance_triples.append((instance, instance_role, concept))

        for triple in filter(lambda x: not is_instance(x), graph.triples):
            self.log.debug('triple: %s', triple)
            source, role, target = triple
            self._add_instance(source, role, True)
            self._add_instance(target, role, False)

    def _add_instance(self, concept, role, is_source):
        if concept in self.instance_by_node:
            return
        elif is_variable(concept):
            self.instance_by_node[concept] = concept
            return
        elif is_amrset_name(concept):
            if role == ':amr-set' and is_source:
                self.instance_by_node[concept] = concept
                return
        elif is_const(concept):
            return
        elif not is_source and role in _roles_with_attrs_at_right:
            self.log.warn('Concept node is generated for the possible attribute '
            + 'name, please use (%s / -) if it is not expected', concept)
        instance = self._get_unique_instance(concept)
        self.instance_by_node[concept] = instance
        self.instance_triples.append((instance, ":instance", concept))

    def _get_unique_instance(self, target):
        if is_variable(target) or is_amrset_name(target):
            return super()._get_unique_instance(target[1:])
        else:
            return super()._get_unique_instance(target)

class ParsedAmr:

    def __init__(self, top, triples):
        self.top = top
        self.triples = triples

    def __iter__(self):
        return self.triples.__iter__()

    def get_top(self):
        return self.top

class TripleProcessor:

    def __init__(self, instance_dict_constr=AmrInstanceDict):
        self.log = logging.getLogger(__name__ + '.TripleProcessor')
        self.next_id = 0
        self.instance_dict_constr = instance_dict_constr

    def _process_relation(self, triple, amr_instances):
        self.log.debug('_process_relation: triple: %s', triple)
        (source, role, target) = triple
        source = amr_instances.map_node_to_instance(source)
        target = amr_instances.map_node_to_instance(target)
        return (source, role, target)

    def add_triple(self, triple):
        source, role, target = triple
        if role == ':instance':
            self._add_variable(source)

    def _add_variable(self, text):
        id = int(text.split('-')[-1])
        if self.next_id <= id:
            self.next_id = id + 1

    def _next_id(self):
        id = self.next_id
        self.next_id += 1
        return id

    def amr_to_triples(self, amr):
        graph = penman.decode(amr, model=_penman_model)
        return self._graph_to_triples(graph)

    def file_to_triples(self, file):
        for graph in penman.iterdecode(file, model=_penman_model):
            for triple in self._graph_to_triples(graph):
                yield triple

    def _graph_to_triples(self, graph):
        sentence_vars = {}
        amr_instances = self.instance_dict_constr(lambda: self._next_id())
        amr_instances.add_graph(graph)

        top = graph.top
        top = amr_instances.map_node_to_instance(top)

        return ParsedAmr(top, self._triples_generator(amr_instances, graph))

    def _triples_generator(self, amr_instances, graph):
        for triple in amr_instances.get_instance_triples():
            yield triple
        for triple in filter(lambda x: not is_instance(x), graph.triples):
            yield self._process_relation(triple, amr_instances)

