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
        {%- if not item.startswith('_') %}
        ~{{ name }}.{{ item }}
        {%- endif -%}
    {%- endfor %}
    {% endif %}
    {% endblock %}
