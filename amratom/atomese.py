from contextlib import contextmanager

from opencog.atomspace import create_child_atomspace
from opencog.utilities import push_default_atomspace, pop_default_atomspace
from opencog.type_constructors import *

from amratom.types import *

TRUE = TruthValue(1, 1)
FALSE = TruthValue(0, 1)

def EVAL(atom, *args, tv=TRUE):
    return EvaluationLink(atom, ListLink(*args), tv=tv)

def TYPED_VAR(name, types):
    return TypedVariableLink(VariableNode(name),
            TypeChoice(*[ TypeNode(t) for t in types ]))

def INST(instance, concept):
    return AmrInstanceLink(amr_value_atom(instance), amr_concept_atom(concept))

def ROLE(role, from_name, to_name):
    return EvaluationLink(
            AmrRole(role),
            ListLink(
                amr_value_atom(from_name),
                amr_value_atom(to_name)))

def is_variable(word):
    return word == '*' or word.startswith('$')

def is_amrset_name(word):
    return word.startswith('@')

def _amr_atom(name, default):
    if is_amrset_name(name):
        return AmrSet(name)
    elif is_variable(name):
        return AmrVariable(name)
    else:
        return default(name)

def amr_value_atom(name):
    return _amr_atom(name, AmrValue)

def amr_concept_atom(name):
    return _amr_atom(name, AmrConcept)

@contextmanager
def child_atomspace(parent):
    """
    Context manager which creates a child atomspace of the passed one.
    """
    atomspace = create_child_atomspace(parent)
    push_default_atomspace(atomspace)
    try:
        yield atomspace
    finally:
        pop_default_atomspace()

@contextmanager
def default_atomspace(atomspace):
    """
    Context manager which makes the passed atomspace the default one.
    """
    push_default_atomspace(atomspace)
    try:
        yield
    finally:
        pop_default_atomspace()

