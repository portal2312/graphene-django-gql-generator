"""Generate GraphQL queries, mutations, and fragments from a schema.

by. MK KIM

Example::
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'project.debug'
    django.setup()

    from gql_generator import GQLGenerator
    from project.schema import schema

    GQLGenerator.codegen(
        schema,
        fragments_file='fragments.graphql',
        queries_file='queries.graphql',
        mutations_file='mutations.graphql',
    )
"""

import os
import re
import string

from graphene.types.definitions import GrapheneObjectType
from graphql import GraphQLList, GraphQLNonNull, GraphQLObjectType, GraphQLScalarType

from project.schema import schema

FRAGMENT_TEMPLATE = string.Template("""
fragment ${fragment} on ${type} {
    ${fields}
}
""")
QUERY_TEMPLATE = string.Template("""
query ${operation_name}(${variables}) {
    ${query_name}(${arguments})
}
""")
QUERY_TEMPLATE_SINGLE = string.Template("""
query ${operation_name}(${variables}) {
    ${query_name}(${arguments}) {
        ...${fragment}
    }
}
""")
QUERY_TEMPLATE_PAGINATION = string.Template("""
query ${operation_name}(
    ${variables}
) {
    ${query_name}(
    ${arguments}
) {
        __typename
        totalCount
        edgeCount
        pageInfo {
            ...PageInfoField
        }
        edges {
            __typename
            cursor
            node {
                ...${fragment}
            }
        }
    }  
}
""")
MUTATION_TEMPLATE = string.Template("""
mutation ${operation_name}(${variables}) {
    ${mutation_name}(${arguments}) {
        ${fields}
    }
}
""")


class GQLGenerator:
    fragment_exclude = [
        'Query',
        '.+TypeConnection$',
        '.+TypeEdge$',
        '.+Payload$',
        'Mutation',
        'DjangoDebugSQL',
    ]
    fragment_name_replace = ('Type.*', 'Field')
    fragment_template = FRAGMENT_TEMPLATE
    query_exclude = ['_debug']
    mutation_exclude = ['_debug']
    mutation_template = MUTATION_TEMPLATE

    def __init__(self, schema):
        self.schema = schema
        self._fragment_exclude_pattern = '|'.join(self.fragment_exclude)
        self._query_exclude_pattern = '|'.join(self.query_exclude)
        self._mutation_exclude_pattern = '|'.join(self.mutation_exclude)

    @property
    def fragment_exclude_pattern(self):
        return self._fragment_exclude_pattern

    @property
    def query_exclude_pattern(self):
        return self._query_exclude_pattern

    @property
    def mutation_exclude_pattern(self):
        return self._mutation_exclude_pattern

    def _scalar_type_to_gql(self, scalar_type):
        if isinstance(scalar_type, GraphQLNonNull):
            name = f'{self._scalar_type_to_gql(scalar_type.of_type)}!'
        elif isinstance(scalar_type, GraphQLList):
            name = f'[{self._scalar_type_to_gql(scalar_type.of_type)}]'
        else:
            name = scalar_type.name
        return name

    def _arguments_to_name_and_scalar_type(self, arguments):
        for name, scalar in arguments.items():
            scalar_type_gql = self._scalar_type_to_gql(scalar.type)
            if scalar.default_value:
                scalar_type_gql = f'{scalar_type_gql} = {scalar.default_value}'
            yield name, scalar_type_gql

    def get_fragments(self):
        for name, instance in self.schema.get_type_map().items():
            if self.fragment_exclude_pattern and re.match(
                self.fragment_exclude_pattern, name
            ):
                continue
            if not isinstance(instance, GrapheneObjectType):
                continue
            if self.fragment_name_replace:
                fragment = re.sub(*self.fragment_name_replace, string=name)
            else:
                fragment = name
            yield self.fragment_template.substitute(
                fragment=fragment,
                type=name,
                fields='\n'.join(f'{field}' for field in instance.fields.keys()),
            )

    def get_queries(self):
        root = self.schema.get_query_type()
        for name, field in root.fields.items():
            if self.query_exclude_pattern and re.match(
                self.query_exclude_pattern, name
            ):
                continue
            if isinstance(field.type, GraphQLScalarType):
                continue
            if isinstance(field.type, GrapheneObjectType):
                field_type = field.type
            else:
                field_type = field.type.of_type

            if isinstance(field_type, GraphQLNonNull):
                fragment = None
                template = QUERY_TEMPLATE
            else:
                fragment = (
                    re.sub(*self.fragment_name_replace, string=field_type.name)
                    if self.fragment_name_replace
                    else field_type.name
                )
                template = (
                    QUERY_TEMPLATE_PAGINATION
                    if 'page' in field.args
                    else QUERY_TEMPLATE_SINGLE
                )

            variables = []
            arguments = []
            for arg_name, arg_type in self._arguments_to_name_and_scalar_type(
                field.args
            ):
                variables.append(f'${arg_name}: {arg_type}')
                arguments.append(f'{arg_name}: ${arg_name}')

            yield template.substitute(
                operation_name=name,
                variables='\n'.join(variables),
                query_name=name,
                arguments='\n'.join(arguments),
                fragment=fragment,
            )

    def _mutation_field_fragment(self, field_type):
        if isinstance(field_type, GraphQLObjectType):
            if field_type.name == 'ErrorType':
                fragment = 'ErrorTypeField'
            elif self.fragment_name_replace:
                fragment = re.sub(*self.fragment_name_replace, string=field_type.name)
            else:
                fragment = field_type.name
        elif isinstance(field_type, GraphQLList):
            if hasattr(field_type, 'of_type'):
                fragment = self._mutation_field_fragment(field_type.of_type)
        elif isinstance(field_type, GraphQLNonNull):
            fragment = None
        else:
            fragment = None
        return fragment

    def _mutation_fields(self, fields):
        for name, field in fields.items():
            fragment = self._mutation_field_fragment(field.type)
            if fragment:
                yield (f'{name} {{', f'...{fragment}', '}')
            else:
                yield (name,)

    def get_mutations(self):
        root = schema.get_mutation_type()
        for name, field in root.fields.items():
            if self.mutation_exclude_pattern and re.match(
                self.mutation_exclude_pattern, name
            ):
                continue
            if not isinstance(field.type, GrapheneObjectType):
                continue
            variables = []
            arguments = []
            for arg_name, arg_type in self._arguments_to_name_and_scalar_type(
                field.args
            ):
                variables.append(f'${arg_name}: {arg_type}')
                arguments.append(f'{arg_name}: ${arg_name}')
            fields = (
                mutation_field
                for mutation_fields in self._mutation_fields(field.type.fields)
                for mutation_field in mutation_fields
                if not mutation_field.startswith('_')
            )
            yield self.mutation_template.substitute(
                operation_name=name,
                variables=', '.join(variables),
                mutation_name=name,
                arguments=', '.join(arguments),
                fields='\n'.join(fields),
            )

    # def get_subscriptions(self):
    #     root = schema.get_subscription_type()

    def save(self, file, handler):
        with open(file, 'w', encoding='utf-8') as f:
            for context in handler():
                f.write(context)
        print('[save]', os.path.abspath(file))

    @classmethod
    def codegen(
        cls,
        schema,
        fragments_file=None,
        mutations_file=None,
        queries_file=None,
        **kwargs,
    ):
        """Code generate."""
        instance = cls(schema, **kwargs)
        if fragments_file:
            instance.save(fragments_file, instance.get_fragments)
        if mutations_file:
            instance.save(mutations_file, instance.get_mutations)
        if queries_file:
            instance.save(queries_file, instance.get_queries)
        print('ok')
