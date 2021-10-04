import logging
import io
import random
import re
from copy import deepcopy
from contextlib import contextmanager

from opencog.atomspace import create_child_atomspace, Atom
from opencog.utilities import push_default_atomspace, pop_default_atomspace
from opencog.type_constructors import *
from opencog.bindlink import execute_atom

from amratom.triples import (TripleProcessor, PatternInstanceDict, AmrInstanceDict,
                     is_amr_set, is_const)
from amratom.atomese import (amr_value_atom, amr_concept_atom, TYPED_VAR, EVAL, TRUE,
                             FALSE, child_atomspace, default_atomspace)
from amratom.types import *

@contextmanager
def child_amr_atomspace(parent):
    """
    Context manager which creates a child AMR atomspace of the passed one.
    """
    amr_space = parent.child()
    push_default_atomspace(amr_space.atomspace)
    try:
        yield amr_space
    finally:
        pop_default_atomspace()

class AmrAtomspace:

    def __init__(self, atomspace):
        self.log = logging.getLogger(__name__ + '.' + type(self).__name__)
        self.atomspace = atomspace

    def child(self):
        atomspace = create_child_atomspace(self.atomspace)
        return AmrAtomspace(atomspace)

    def add_triple(self, triple):
        with default_atomspace(self.atomspace):
            self.log.debug('add_triple: %s', triple)
            source, role, target = triple
            top_atom = amr_value_atom(source)
            if role == ':instance':
                AmrInstanceLink(top_atom, amr_concept_atom(target))
            else:
                if role.endswith("?"):
                    role = role[:-1]
                    optional = True
                else:
                    optional = False
                role_atom = AmrRole(role)
                EvaluationLink(role_atom,
                        ListLink(
                            top_atom,
                            amr_value_atom(target)))
                EvaluationLink(PredicateNode("is-optional"),
                        ListLink(top_atom, role_atom, amr_value_atom(target)),
                                 tv=TRUE if optional else FALSE)
                source_amr_concept = self.get_concepts_of_instance(top_atom)
                if len(source_amr_concept) > 0 and source_amr_concept[0].is_a(types.AmrConcept):
                    EvaluationLink(PredicateNode("has-role"),
                            ListLink(source_amr_concept[0], role_atom))

    def get_concept_instances(self, amr_concept):
        with child_atomspace(self.atomspace) as atomspace:
            # We don't apply type constraints on VariableNode("instance") here.
            # Any type-checking should be used outside (e.g. one may want to
            # detect unexpected types as KB errors)
            # TODO? variable type can be added as an optional argument once needed
            return execute_atom(atomspace,
                GetLink(AmrInstanceLink(VariableNode("instance"), amr_concept))).out

    def get_concepts_of_instance(self, amr_instance):
        with child_atomspace(self.atomspace) as atomspace:
            results = execute_atom(atomspace,
                GetLink(AmrInstanceLink(amr_instance, VariableNode("concept")))).out
            if len(results) > 1:
                self.log.debug('get_concept: WARNING - multimple AmrInstanceLinks for %s', amr_instance.name)
            return results

    def get_relations(self, pred, arg0, arg1, var_types=None):
        with child_atomspace(self.atomspace) as atomspace:
            if var_types is not None:
                vars = []
                for arg, types in var_types.items():
                    if types is None:
                        vars.append(VariableNode(arg))
                    else:
                        types = types if isinstance(types, list) else [ types ]
                        vars.append(TYPED_VAR(arg, types))
                results = execute_atom(atomspace,
                    GetLink(VariableList(*vars),
                        EvaluationLink(pred, ListLink(arg0, arg1))))
            else:
                results = execute_atom(atomspace,
                    GetLink(EvaluationLink(pred, ListLink(arg0, arg1))))
            return [r.out if r.is_link() else r for r in results.out]

    def get_concept_roles(self, arg0, arg1, var_types=None):
        return self.get_relations(PredicateNode("has-role"), arg0, arg1, var_types)

    def get_instance_roles(self, amr_instance):
        return [self.get_relations(VariableNode("role"), amr_instance, VariableNode("right-inst"),
                                   {"role": "AmrRole", "right-inst": None}),
                self.get_relations(VariableNode("role"), VariableNode("left-inst"), amr_instance,
                                   {"role": "AmrRole", "left-inst": None})]

    def get_amrsets_by_concept(self, concept):
        with child_atomspace(self.atomspace) as atomspace:
            amrsets = []
            results = execute_atom(atomspace, GetLink(
                VariableList(
                    TypedVariableLink(VariableNode("amrset"),
                        TypeNode("AmrSet")),
                    TypedVariableLink(VariableNode("amrset-instance"),
                            TypeNode("AmrValue")),
                    VariableNode("concept")),
                AndLink(
                    EvaluationLink(AmrRole(":amr-set"),
                                   ListLink(VariableNode("amrset"),
                                       VariableNode("amrset-instance"))),
                    AmrInstanceLink(VariableNode("amrset-instance"),
                        VariableNode("concept"))
                )))
            for x in results.out:
                amrset, amrset_instance, concept_candidate = x.out
                if concept_candidate.is_a(types.AmrVariable):
                    amrsets.append((amrset, amrset_instance))
                elif concept_candidate.is_a(types.AmrSet):
                    amrsets.append((amrset, amrset_instance))
                elif concept is not None and match_concept(concept, concept_candidate):
                    amrsets.append((amrset, amrset_instance))

            return amrsets

    def get_concept(self, concept):
        with child_atomspace(self.atomspace) as atomspace:
            results = execute_atom(atomspace, GetLink(
                VariableList(
                    TypedVariableLink(VariableNode("parent"),
                        TypeChoice(
                            TypeNode("AmrConcept"),
                            TypeNode("AmrVariable"),
                            TypeNode("AmrSet")))),
                AndLink(AmrInstanceLink(concept, VariableNode("parent")))
                ))
            return results.out[0] if len(results.out) > 0 else None

_meaning_postfix_pattern = re.compile(r'-\d+$')

def match_concept(input, template):
    if _meaning_postfix_pattern.search(template.name) is not None:
        # the template specifies an exact meaning
        return input.name == template.name
    else:
        meaning_pos = _meaning_postfix_pattern.search(input.name)
        if meaning_pos is None:
            return input.name == template.name
        else:
            return input.name[:meaning_pos.start(0)] == template.name

class AmrMatch:

    def __init__(self, amrset, vars={}):
        self.amrset = amrset
        self.vars = vars

    def __repr__(self):
        return ('{ amrset: ' + repr(self.amrset) + ', vars: ' +
            repr(self.vars) + ' }')

    def __eq__(self, other):
        return self.amrset == other.amrset and self.vars == other.vars

class AmrMatcher:
    def __init__(self, space):
        self.log = logging.getLogger(__name__ + '.' + type(self).__name__)
        self.space = space

    def match_amr_set(self, input_value, template_value, amrset):
        self.log.debug("match_amr_set: input_value: %s, template_value: %s, amrset: %s",
                       input_value, template_value, amrset)
        matches = []
        for candidate in self.space.get_relations(AmrRole(":amr-set"), amrset, VariableNode("target")):
            for match in self.match_amr_trees(input_value, candidate,
                    amrset_instance=template_value):
                matches.append({ amrset: match })
        return matches

    def match_amr_trees(self, input_value, template_value, amrset_instance=None):
        # match values
        self.log.debug('match_amr_trees: input_value: %s, template_value: %s'
                + ', amrset_instance: %s', input_value, template_value,
                amrset_instance)
        if (input_value == template_value):
            return [{}]
        if template_value.is_a(types.AmrVariable):
            matches = list(self.match_value(input_value))
            if len(matches) == 0:
                # instance AmrVariable
                return [{ template_value: input_value }]
            else:
                result = []
                for match in matches:
                    result.append({ template_value: { match.amrset: match.vars } })
                return result

        # match concepts
        input_concept = self.space.get_concept(input_value)
        template_concept = self.space.get_concept(template_value)
        self.log.debug('match_amr_trees: input_concept: %s template_concept: %s',
                       input_concept, template_concept)
        match = {}
        if (input_concept is None and template_concept is None):
            self.log.debug('match_amr_trees: different attributes')
            return []
        elif template_concept is None:
            self.log.debug('match_amr_trees: template concept is None and input concept is not')
            return []
        elif input_concept is None:
            self.log.debug('match_amr_trees: input concept is None and template concept is not')
            return []
        elif template_concept.is_a(types.AmrSet):
            # hierarchical template
            return self.match_amr_set(input_value, template_value,
                    template_concept)
        elif template_concept.is_a(types.AmrVariable):
            # parent AnchorNode
            match[template_concept] = input_concept
        elif not match_concept(input_concept, template_concept):
            self.log.debug('match_amr_trees: different concepts')
            return []

        # match roles
        return self.match_amr_roles(input_value, template_value, match,
                amrset_instance)

    class RoleMetadata:
        def __init__(self, role):
            self.role = role
            self.targets = []

    def match_amr_roles(self, input_value, template_value, match,
            amrset_instance=None):
        self.log.debug('match_amr_roles: input_value: %s, template_value: %s'
                + ', amrset_instance: %s', input_value, template_value,
                amrset_instance)

        input_roles = {}
        for role, target in self.space.get_relations(VariableNode("role"),
                input_value, VariableNode("target"),
                { "role": "AmrRole", "target": None }):
            if role not in input_roles:
                input_roles[role] = set()
            input_roles[role].add(target)

        template_roles = {}
        for role, target in self.space.get_relations(VariableNode("role"),
                template_value, VariableNode("target"),
                { "role": "AmrRole", "target": None }):
            if role not in template_roles:
                template_roles[role] = self.RoleMetadata(role)
            template_roles[role].targets.append((template_value, target))

        if amrset_instance is not None:
            for role, target in self.space.get_relations(VariableNode("role"),
                    amrset_instance, VariableNode("target"),
                    { "role": "AmrRole", "target": None }):
                if role.name == ':amr-set':
                    continue
                if role not in template_roles:
                    template_roles[role] = self.RoleMetadata(role)
                template_roles[role].targets.append((amrset_instance, target))

        matches = [ match ]
        absent_input_roles = set()
        absent_template_roles = set(template_roles.keys())
        has_role_wildcard = AmrRole(":*") in template_roles
        for role in input_roles:
            if role in template_roles:
                absent_template_roles.remove(role)
            else:
                if role.name == ':pos' or has_role_wildcard:
                    continue
                else:
                    absent_input_roles.add(role)
                    continue

            for next_input_value in input_roles[role]:
                for source, next_template_value in template_roles[role].targets:
                    new_matches = []
                    for res in self.match_amr_trees(next_input_value,
                            next_template_value):
                        for prev_match in matches:
                            new_match = prev_match.copy()
                            new_match.update(res)
                            new_matches.append(new_match)
                    # Here we stop on the first possible match for the role.
                    # There are may be other options to match role targets in
                    # other sequence, but we ignore this for now.
                    if len(new_matches) > 0:
                        template_roles[role].targets.remove((source, next_template_value))
                        break
                matches = new_matches
                if len(matches) == 0:
                    self.log.debug('match_amr_roles: no match input for role: '
                            + '%s, value: %s, template_targets: %s',
                            role, next_input_value, template_roles[role].targets)
                    return []

            absent_mandatory_roles = self.get_mandatory_roles(template_roles[role])
            if len(absent_mandatory_roles) > 0:
                self.log.debug("match_amr_roles: non optional template roles are " +
                        "absent in input_value: %s", absent_mandatory_roles)
                return []

        if len(absent_input_roles) > 0:
            self.log.debug("match_amr_roles: input_value has roles which are " +
                    "not present in template_value: %s", absent_input_roles)
            return []

        for role in absent_template_roles:
            absent_mandatory_roles = self.get_mandatory_roles(template_roles[role])
            if len(absent_mandatory_roles) > 0:
                self.log.debug("match_amr_roles: non optional template roles are " +
                        "absent in input_value: %s", absent_mandatory_roles)
                return []

        self.log.debug('match_amr_roles: matches found, vars: %s', matches)
        return matches

    def get_mandatory_roles(self, role_metadata):
        mandatory_roles = []
        for source, target in role_metadata.targets:
            optional = self.is_optional_role(source, role_metadata.role, target)
            if not optional:
                mandatory_roles.append((role_metadata.role, source, target))
        return mandatory_roles

    def is_optional_role(self, template_value, role, target):
        return (role == AmrRole(":*") or
                EvaluationLink(PredicateNode("is-optional"),
                    ListLink(template_value, role, target)).tv == TRUE)

    def match_value(self, value):
        concept = self.space.get_concept(value)
        self.log.debug("match_value: value: %s, concept: %s", value, concept)
        for amrset, amrset_var in self.space.get_amrsets_by_concept(concept):
            self.log.debug('match_value: try amrset: %s, instance: %s', amrset, amrset_var)
            for match in self.match_amr_trees(value, amrset_var):
                self.log.debug("match_value: found match: %s", match)
                yield AmrMatch(amrset, match)


_single_word_pattern = re.compile(r'^\S+$')

class AmrTemplateInstance:

    def __init__(self, match=None):
        self.vars = {}
        self.subint = []
        if match:
            self.amrset = match.amrset.name
            self.subint, self.vars = self._unwrap_vars_rec(match.vars)

    def _unwrap_vars_rec(self, vs):
        subint = []
        vacc = {}
        for key, value in vs.items():
            if isinstance(key, Atom) and key.name != '*':
                if key.is_a(types.AmrSet):
                    subint_child, vacc_child = self._unwrap_vars_rec(value)
                    subint += [key.name] + subint_child
                    # Storing all subintent variables as a value for subintent
                    if len(vacc_child) > 0:
                        vacc[key.name] = vacc_child
                    # And also duplicate them in a flatten dict for convenience
                    vacc.update(vacc_child)
                else:
                    assert key.is_a(types.AmrVariable), "Unsupported variable {0}".format(key)
                    vname = None
                    if isinstance(value, Atom):
                        vname = value.name
                        if vname.startswith('"') and vname.endswith('"'):
                            vname = vname[1:-1]
                    elif isinstance(value, dict):
                        vkey = list(value.keys())[0]
                        assert len(value) == 1 and isinstance(vkey, Atom) and vkey.is_a(types.AmrSet), \
                            "Expected only one AmrSet as variable value: {0} - {1}".format(key, value)
                        vname = vkey.name
                        subint_child, vacc_child = self._unwrap_vars_rec(value[vkey])
                        subint += [vname] + subint_child
                        if len(vacc_child) > 0:
                            vname = {vname: vacc_child}
                        vacc.update(vacc_child)
                    vacc[key.name] = vname
        return subint, vacc

    @classmethod
    def from_values(cls, amrset, vars={}, subint=[]):
        inst = cls()
        inst.amrset = amrset
        inst.vars = deepcopy(vars)
        inst.subint = subint
        return inst

    def __repr__(self):
        return ('{ amrset: ' + self.amrset + ', vars: ' +
            repr(self.vars) + ' }' + ', subint: ' + repr(self.subint))

    def __eq__(self, other):
        return self.amrset == other.amrset and self.vars == other.vars and \
            self.subint == other.subint


class PatternParser:

    def __init__(self, amr_proc, amr_space):
        self.log = logging.getLogger(__name__ + '.' + type(self).__name__)
        self.amr_proc = amr_proc
        self.amr_space = amr_space
        self.triple_proc = TripleProcessor(PatternInstanceDict)
        self.utterance_parser = UtteranceParser(amr_proc, amr_space)

    def parse(self, amr):
        with default_atomspace(self.amr_space.atomspace):
            parsed_amr = self.triple_proc.amr_to_triples(amr)
            self._process_triples(parsed_amr)
            return amr_value_atom(parsed_amr.top)

    def load_file(self, file):
        self._process_triples(self.triple_proc.file_to_triples(file))

    def load_text(self, text):
        with io.StringIO(text) as file:
            self.load_file(file)

    def _process_triples(self, triples):
        for triple in triples:
            if is_amr_set(triple) and is_const(triple[2]):
                source, role, target = triple
                no_quotes = target[1:-1]
                if _single_word_pattern.search(no_quotes) is not None:
                    self.amr_space.add_triple(triple)
                else:
                    top = self.utterance_parser.parse_sentence(no_quotes)
                    self.amr_space.add_triple((source, role, top.name))
            else:
                self.amr_space.add_triple(triple)


class UtteranceParser:

    def __init__(self, amr_proc, amr_space):
        self.log = logging.getLogger(__name__ + '.' + type(self).__name__)
        self.amr_proc = amr_proc
        self.amr_space = amr_space
        self.triple_proc = TripleProcessor(AmrInstanceDict)
        # FIXME: NB: to have unique varible names we need importing all
        # triples into triple_proc before processing
        self.triple_proc.next_id = 500000;

    def parse(self, text):
        with default_atomspace(self.amr_space.atomspace):
            sentences = []
            amrs = self.amr_proc.utterance_to_amr(text)
            for amr in amrs:
                parsed_amr = self.triple_proc.amr_to_triples(amr)
                for triple in parsed_amr:
                    self.amr_space.add_triple(triple)
                sentences.append(amr_value_atom(parsed_amr.top))
            return sentences

    def parse_sentence(self, text):
        sentences = self.parse(text)
        assert len(sentences) == 1, 'Single sentence is expected as input'
        return sentences[0]

class AmrGenerator:

    def __init__(self, amr_space, amr_proc):
        self.log = logging.getLogger(__name__ + '.' + type(self).__name__)
        self.amr_space = amr_space
        self.amr_proc = amr_proc

    def recSubst(self, atom):
        bSubst = True
        with child_atomspace(self.amr_space.atomspace) as atomspace:
            while bSubst and atom:
                bSubst = False
                if atom.is_a(types.AmrVariable):
                    if atom.name == '*':
                        # Just ignore wildcards since we can't guess their content
                        self.log.debug("recSubst: ignore *")
                        atom = None
                    else:
                        subst = execute_atom(atomspace,
                            GetLink(StateLink(atom, VariableNode("$state")))).out
                        for s in subst:
                            if not s.is_a(types.VariableNode):
                                if bSubst:
                                    self.log.debug('recSubst: WARNING - additional value %s', s.name)
                                else:
                                    self.log.debug('recSubst: AmrVariable %s --> %s', atom.name, s.name)
                                atom = s
                                bSubst = True
                        if not bSubst:
                            self.log.debug('recSubst: WARNING - no value for AmrVariable %s', atom.name)
                else:
                    # Check if atom refers to AmrSet (set of tops of amr graphs)
                    amr_set = execute_atom(atomspace,
                        GetLink(EVAL(AmrRole(":amr-set"), atom, VariableNode("$top")))).out
                    if not atom.is_a(types.AmrSet) and len(amr_set) > 0:
                        self.log.debug('recSubst: WARNING - non-AmrSet atom %s has :amr-set role', atom.name)
                    if atom.is_a(types.AmrSet) and len(amr_set) == 0:
                        self.log.debug('recSubst: WARNING - AmrSet atom %s has no :amr-set role', atom.name)
                    if len(amr_set) > 0:
                        # Just choose randomly now. A more advanced strategy could be
                        # to filter out subgraphs from amr-set with AmrVariable lacking StateLinks
                        s = amr_set[random.randint(0, len(amr_set)-1)]
                        self.log.debug('recSubst: :amr-set substitution %s --> %s', atom.name, s.name)
                        atom = s
                        bSubst = True
        return atom

    def triplesFromFullGraph(self, topAtom):
        self.log.debug('triplesFromFullGraph: generating %s', topAtom.name if topAtom else None)
        topAtom = self.recSubst(topAtom)
        triples = []
        if not topAtom: return triples
        parentName = parent = None
        # identify parent name with substitutions
        if topAtom.is_a(types.AmrValue):
            pa = self.amr_space.get_concepts_of_instance(topAtom)
            if len(pa) > 0:
                # If the parent explicitly refers to an AmrSet, AmrSet will
                # be sampled, and AmrValue will appear as a parent.
                # It is difficult to track such situations in case of other
                # recursive substitutions (e.g. with variables), so we simply assume
                # that the situation of an AmrValue as the parent is valid and implies
                # the necessity to merge two graphs referring to these AmrValues.
                parent = self.recSubst(pa[0])
                parentName = parent.name
        children, _ = self.amr_space.get_instance_roles(topAtom)
        connections = []
        for child in children:
            self.log.debug('triplesFromFullGraph: child %s %s %s', topAtom.name, child[0].name, child[1].name)
            # Fixme? recSubst will be called second time for topAtom in recursion
            # It's not a huge problem since it will do nothing, although it will generate warnings twice
            atom2 = self.recSubst(child[1])
            if not atom2 or atom2.is_a(types.VariableNode) or atom2.is_a(types.AmrVariable):
                # TODO? Maybe, we need raise an exception for a non-optional role
                continue
            if child[0].name == ':pos':
                # we consider :pos connected to constant attributes only now
                if parentName:
                    parentName += "~"+atom2.name.replace("\"", "")
                else:
                    self.log.debug('triplesFromFullGraph: WARNING - cannot add pos tag to %s', topAtom.name)
            elif child[0].name == ':*':
                self.log.debug('triplesFromFullGraph: ignoring :*')
                continue
            else:
                # We don't consider optional roles here. They are represented by PredicateNode("is-optional")
                # which is ignored here, and full graph is reconstructed.
                # Controlling how to deal with optional roles in the generator can be added in future.
                connections += [(topAtom.name, child[0].name, atom2)]
        if parentName:
            self.log.debug('triplesFromFullGraph: topAtom %s / %s', topAtom.name, parentName)
            if parent.is_a(types.AmrConcept):
                # topAtom is just an instance of AmrConcept
                triples += [(topAtom.name, ":instance", parentName)]
            elif parent.is_a(types.AmrVariable):
                assert False, "AmrVariable {0} is not set".format(parent.name)
            else:
                assert parent.is_a(types.AmrValue), "Unexpected type of {0} after recSubst".format(parent.name)
                # Generating subgraph to be merged
                side_triples = self.triplesFromFullGraph(parent)
                if len(side_triples) > 0:
                    for triple in side_triples:
                        if triple[0] == parentName:
                            # Substituting current top in place of the top of the graph to be merged
                            triples += [(topAtom.name, triple[1], triple[2])]
                        else:
                            triples += [triple]
        # In case of (@game-name :amr-set "Bingo"), generateFull(AmrSet("@game-name")) will
        # return an empty graph, because there are no triples. Generating single attribute value
        # is not supported.
        # elif len(connections) == 0:
        #     triples += [(topAtom.name, ':instance', None)]
        # Just sorting alphabetically works reasonably well: :ARG0,1,2 go first
        connections = sorted(connections, key = lambda n: n[1])
        for c in connections:
            child_triple = self.triplesFromFullGraph(c[2])
            if len(child_triple) == 0:
                # The special case of amr attribute that substitutes amr value.
                # E.g. for (@game :amr-set "Bingo"), (name :op1 @game), we will have
                # n / name pointing to g / @game, so we need to peek into g's parent
                # to understand that we need not (n :op1 g) with triples for g,
                # but (n :op1 "Bingo"), where "Bingo" is recSubst of g's parent.
                parent2 = self.amr_space.get_concepts_of_instance(c[2])
                if len(parent2) > 0:
                    parent2 = self.recSubst(parent2[0])
                    if not parent2 is None:
                        triples += [(c[0], c[1], parent2.name)]
                        continue
            new_triples = [(c[0], c[1], c[2].name)] + child_triple
            for tn in new_triples:
                isNew = True
                for tp in triples:
                    if tp[0] == tn[0] and tp[1] == tn[1]:
                        isNew = False
                        if tp[2] != tn[2] or tp[1] != ':instance':
                            self.log.debug('triplesFromFullGraph: WARNING - conflicting (%s %s %s) (%s %s %s)',
                                           tp[0], tp[1], tp[2], tn[0], tn[1], tn[2])
                            if tp[2] != tn[2] and tp[1] != ':instance':
                                # "Do you have any other questions for me?" contains
                                # `:mod (o / other)` and `:mod (a2 / any)` simultaneously
                                isNew = True
                        # else do nothing - it is expected for :instance
                if isNew:
                    triples += [tn]
        return triples

    def renameInsts(self, triples):
        names = [t[0] for t in triples] + [t[2] for t in triples]
        for i in range(len(triples)):
            t = triples[i]
            if t[1] != ':instance': continue
            oldName = t[0]
            newName = t[2][0] # oldName[0] also works, but it can produce w/i, when what-0015/$what is in graph
            while newName in names:
                if len(newName) == 1:
                    newName += newName[0]
                elif len(newName) == 2:
                    newName += 'a'
                else:
                    newName = newName[0:2] + chr(ord(newName[2])+1)
            for j in range(len(triples)):
                if triples[j][0] == oldName: triples[j] = (newName, triples[j][1], triples[j][2])
                if triples[j][2] == oldName: triples[j] = (triples[j][0], triples[j][1], newName)
            names += [newName]
        return triples

    def generateFull(self, topAtom):
        triples = self.triplesFromFullGraph(topAtom)
        r = self.renameInsts(triples)
        text = self.amr_proc.triples_to_utterance(r) if r != [] else None
        if text is None:
            self.log.debug('generateFull: WARNING - No graph for topAtom %s', topAtom.name)
        return text

