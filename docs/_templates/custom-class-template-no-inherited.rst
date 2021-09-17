{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block members %}
    {% if members %}
    .. rubric:: {{ _('Methods & Attributes') }}

    .. autosummary::
        :toctree:
        :caption: Methods & Attributes
    {% for item in members %}
        {%- if not item.startswith('_') and item not in inherited_members %}
        ~{{ name }}.{{ item }}
        {%- endif -%}
    {%- endfor %}
    {% endif %}
    {% endblock %}
