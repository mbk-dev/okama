# from typing import Union, Callable, List, TypeVar, Any
#
# FuncType = Callable[..., Any]
# F = TypeVar("F", bound=FuncType)
#
#
# def doc(*docstrings: Union[str, Callable], **params) -> Callable[[F], F]:
#     """
#     A decorator take docstring templates, concatenate them and perform string
#     substitution on it.
#
#     This decorator will add a variable "_docstring_components" to the wrapped
#     callable to keep track the original docstring template for potential usage.
#     If it should be consider as a template, it will be saved as a string.
#     Otherwise, it will be saved as callable, and later user __doc__ and dedent
#     to get docstring.
#
#     Parameters
#     ----------
#     *docstrings : str or callable
#         The string / docstring / docstring template to be appended in order
#         after default docstring under callable.
#     **params
#         The string which would be used to format docstring template.
#     """
#
#     def decorator(decorated: F) -> F:
#         # collecting docstring and docstring templates
#         docstring_components: List[Union[str, Callable]] = []
#         if decorated.__doc__:
#             docstring_components.append(dedent(decorated.__doc__))
#
#         for docstring in docstrings:
#             if hasattr(docstring, "_docstring_components"):
#                 # error: Item "str" of "Union[str, Callable[..., Any]]" has no
#                 # attribute "_docstring_components"  [union-attr]
#                 # error: Item "function" of "Union[str, Callable[..., Any]]"
#                 # has no attribute "_docstring_components"  [union-attr]
#                 docstring_components.extend(
#                     docstring._docstring_components  # type: ignore[union-attr]
#                 )
#             elif isinstance(docstring, str) or docstring.__doc__:
#                 docstring_components.append(docstring)
#
#         # formatting templates and concatenating docstring
#         decorated.__doc__ = "".join(
#             [
#                 component.format(**params)
#                 if isinstance(component, str)
#                 else dedent(component.__doc__ or "")
#                 for component in docstring_components
#             ]
#         )
#
#         # error: "F" has no attribute "_docstring_components"
#         decorated._docstring_components = (  # type: ignore[attr-defined]
#             docstring_components
#         )
#         return decorated
#
#     return decorator
#
#
# def dedent(text):
#     """Remove any common leading whitespace from every line in `text`.
#
#     This can be used to make triple-quoted strings line up with the left
#     edge of the display, while still presenting them in the source code
#     in indented form.
#
#     Note that tabs and spaces are both treated as whitespace, but they
#     are not equal: the lines "  hello" and "\\thello" are
#     considered to have no common leading whitespace.
#
#     Entirely blank lines are normalized to a newline character.
#     """
#     # Look for the longest leading string of spaces and tabs common to
#     # all lines.
#     margin = None
#     text = _whitespace_only_re.sub('', text)
#     indents = _leading_whitespace_re.findall(text)
#     for indent in indents:
#         if margin is None:
#             margin = indent
#
#         # Current line more deeply indented than previous winner:
#         # no change (previous winner is still on top).
#         elif indent.startswith(margin):
#             pass
#
#         # Current line consistent with and no deeper than previous winner:
#         # it's the new winner.
#         elif margin.startswith(indent):
#             margin = indent
#
#         # Find the largest common whitespace between current line and previous
#         # winner.
#         else:
#             for i, (x, y) in enumerate(zip(margin, indent)):
#                 if x != y:
#                     margin = margin[:i]
#                     break
#
#     # sanity check (testing/debugging only)
#     if 0 and margin:
#         for line in text.split("\n"):
#             assert not line or line.startswith(margin), \
#                    "line = %r, margin = %r" % (line, margin)
#
#     if margin:
#         text = re.sub(r'(?m)^' + margin, '', text)
#     return text
