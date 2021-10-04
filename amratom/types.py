from opencog.atomspace import types, decl_type, type_decl_context
from opencog.utilities import add_node, add_link

with type_decl_context(__name__):
    decl_type(types.Node, 'AmrConcept')
    decl_type(types.Node, 'AmrValue')
    decl_type(types.Node, 'AmrVariable')
    decl_type(types.Node, 'AmrSet')
    decl_type(types.PredicateNode, 'AmrRole')
    decl_type(types.OrderedLink, 'AmrInstanceLink')

def AmrConcept(name, tv=None):
    return add_node(types.AmrConcept, name, tv)

def AmrValue(name, tv=None):
    return add_node(types.AmrValue, name, tv)

def AmrVariable(name, tv=None):
    return add_node(types.AmrVariable, name, tv)

def AmrSet(name, tv=None):
    return add_node(types.AmrSet, name, tv)

def AmrRole(name, tv=None):
    return add_node(types.AmrRole, name, tv)

def AmrInstanceLink(*args, tv=None):
    return add_link(types.AmrInstanceLink, args, tv)

