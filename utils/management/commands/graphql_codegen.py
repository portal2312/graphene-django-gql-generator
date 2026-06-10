"""Django app 별로 GraphQL operation 을 추출하는 Django management command.

Self-contained — 전체 :class:`GQLGenerator` 를 직접 embed 하므로 이 command 는
``utils.gql_generator`` 에 의존하지 않는다. Generator 는 Graphene/GraphQL-core
schema 를 introspection 하여 Django app 마다 하위 directory 하나를 만들고, 그
안에 app 별 ``.graphql`` file (``fragments.graphql``, ``queries.graphql``,
``mutations.graphql``, ``subscriptions.graphql``) 을 작성한다.
"""

from __future__ import annotations

import importlib

from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser
from graphene.utils.str_converters import to_camel_case
from graphql import (
    GraphQLArgument,
    GraphQLField,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
)
from graphql.pyutils import Undefined

#: 등록된 app 에 귀속할 수 없는 ObjectType 의 내부 grouping key (예: framework
#: wrapper, ``PageInfo``, JWT response type). 이 type 들은 app 별 하위 directory
#: 가 아니라 출력 root 의 :data:`GLOBAL_FILENAME` 에 직접 출력된다.
SHARED_LABEL: str = "_shared"

#: 소유 Django app 이 없는 shared / framework ObjectType 의 fragment 를 담는
#: 최상위 fragments file 의 이름.
GLOBAL_FILENAME: str = "global.graphql"

#: 출력 root 에 출력되는 legacy aggregate file (``fragments.graphql``,
#: ``queries.graphql``, ``mutations.graphql``, ``subscriptions.graphql``) 의
#: 내부 summary key. App 별 분할과 함께 유지되어 호출자가 완전성을 검증할 수
#: 있게 한다 — aggregate 에는 있으나 app 별 출력에 없는 것은 어느 등록 app 에도
#: 귀속할 수 없었던 field/type 이다.
AGGREGATE_LABEL: str = "_aggregate"


@dataclass(frozen=True)
class AppSpec:
    """단일 Django app 에 대한 codegen partition 명세.

    Attributes:
        label: 출력 하위 directory 이름 (보통 ``Model._meta.app_label``).
        module_prefix: Python module path prefix. Django 가 아닌
            ``graphene.ObjectType`` 의 ``__module__`` 이 이 app 아래에 있을 때
            소유 app 을 판별하는 데 쓴다.
        schema_modules: 완전한 형태로 명시한 schema module 이름 목록. 단일
            병합 ``schema`` module 을 publish 하는 app 에 쓴다
            (예: ``pprms.schema``).
        discover_under: ``<sub>.schema`` module 을 찾으려고 walk 할 package.
            GraphQL 정의를 여러 sub-package 로 나눈 app 에 쓴다
            (예: ``apps.common.graphql.system_status.schema``).
    """

    label: str
    module_prefix: str
    schema_modules: tuple[str, ...] = ()
    discover_under: str | None = None

    def iter_schema_modules(self) -> Iterator[str]:
        """이 app 이 소유한 모든 schema module 이름을 yield 한다."""
        yield from self.schema_modules
        if self.discover_under:
            try:
                pkg = importlib.import_module(self.discover_under)
            except ImportError:
                return
            pkg_file = getattr(pkg, "__file__", None)
            if not pkg_file:
                return
            for entry in sorted(Path(pkg_file).parent.iterdir()):
                if entry.is_dir() and (entry / "schema.py").exists():
                    yield f"{self.discover_under}.{entry.name}.schema"


#: 프로젝트 기본 app registry. CLI 출력에서 순서가 보존된다.
DEFAULT_APP_REGISTRY: dict[str, AppSpec] = {
    "common": AppSpec(
        label="common",
        module_prefix="apps.common",
        discover_under="apps.common.graphql",
    ),
    "account": AppSpec(
        label="account",
        module_prefix="apps.account",
        discover_under="apps.account.graphql",
    ),
    "pprms": AppSpec(
        label="pprms",
        module_prefix="pprms",
        schema_modules=("pprms.schema",),
    ),
    "pprms_ext": AppSpec(
        label="pprms_ext",
        module_prefix="pprms_ext",
        schema_modules=("pprms_ext.schema",),
    ),
}


class GQLGenerator:
    """Django app 단위로 분할하는 동적 GraphQL operation generator.

    Schema 의 root operation type 들을 walk 하여 도달 가능한 모든
    :class:`GraphQLObjectType` 를 재귀적으로 수집한다. 각 ObjectType 은
    (``DjangoObjectType._meta.model._meta.app_label`` 또는 ``__module__``
    prefix 매칭으로) 소유 Django app 에 귀속된다. Fragment 는 소유 app 의
    ``fragments.graphql`` 에 출력하고, root operation 은 각 app 의 local
    ``Query`` / ``Mutation`` / ``Subscription`` class 를 검사하여 분할한다.
    """

    #: Type 이름에 붙여 fragment 이름을 만드는 suffix.
    fragment_suffix: str = "Fragment"

    #: 렌더링된 GraphQL 에서 쓰는 indent 단위.
    indent: str = "  "

    #: fragment 로 결코 출력하지 않는 type 이름.
    excluded_type_names: frozenset[str] = frozenset(
        {"Query", "Mutation", "Subscription"},
    )

    #: 결코 출력하지 않는 type 이름 prefix (introspection, debug helper).
    excluded_type_prefixes: tuple[str, ...] = ("__", "DjangoDebug")

    #: fragment / operation 렌더링에서 건너뛰는 field 이름 prefix.
    excluded_field_prefixes: tuple[str, ...] = ("_",)

    #: cycle 를 끊을 scalar/enum field 가 없을 때 출력하는 selection.
    #: ``__typename`` 은 어떤 object type 에서도 항상 query 가능하다.
    cycle_break_fallback: str = "__typename"

    #: operation kind 별 app 파일 이름.
    OPERATION_FILENAMES: dict[str, str] = {
        "query": "queries.graphql",
        "mutation": "mutations.graphql",
        "subscription": "subscriptions.graphql",
    }

    #: app 별 fragments 파일 이름.
    FRAGMENT_FILENAME: str = "fragments.graphql"

    def __init__(  # noqa: PLR0913
        self,
        schema: Any,
        *,
        registry: dict[str, AppSpec] | None = None,
        fragment_suffix: str | None = None,
        excluded_type_names: Iterable[str] | None = None,
        excluded_type_prefixes: tuple[str, ...] | None = None,
        excluded_field_prefixes: tuple[str, ...] | None = None,
    ) -> None:
        """Generator 를 초기화한다.

        Args:
            schema: ``graphene.Schema`` *또는* raw ``graphql.GraphQLSchema``.
            registry: app label → :class:`AppSpec` 매핑. 기본값은
                :data:`DEFAULT_APP_REGISTRY`.
            fragment_suffix: :attr:`fragment_suffix` override.
            excluded_type_names: :attr:`excluded_type_names` override.
            excluded_type_prefixes: :attr:`excluded_type_prefixes` override.
            excluded_field_prefixes: :attr:`excluded_field_prefixes` override.
        """
        gql_schema = getattr(schema, "graphql_schema", schema)
        if not isinstance(gql_schema, GraphQLSchema):
            msg = (
                f"Expected graphene.Schema or graphql.GraphQLSchema, "
                f"got {type(schema).__name__}"
            )
            raise TypeError(msg)
        self.schema: GraphQLSchema = gql_schema
        self.registry: dict[str, AppSpec] = registry or DEFAULT_APP_REGISTRY

        if fragment_suffix is not None:
            self.fragment_suffix = fragment_suffix
        if excluded_type_names is not None:
            self.excluded_type_names = frozenset(excluded_type_names)
        if excluded_type_prefixes is not None:
            self.excluded_type_prefixes = tuple(excluded_type_prefixes)
        if excluded_field_prefixes is not None:
            self.excluded_field_prefixes = tuple(excluded_field_prefixes)

    # ──────────────────────────────────────────────────────────────────
    # Type system helpers
    # ──────────────────────────────────────────────────────────────────

    def _unwrap(self, gql_type: Any) -> Any:
        """``NonNull`` / ``List`` wrapper 를 벗기고 최내곽 named type 을 반환한다."""
        while isinstance(gql_type, (GraphQLNonNull, GraphQLList)):
            gql_type = gql_type.of_type
        return gql_type

    def _render_type_ref(self, gql_type: Any) -> str:
        """Wrapping 된 type 을 GraphQL 문법으로 렌더링한다 (예: ``[ID!]!``)."""
        if isinstance(gql_type, GraphQLNonNull):
            return f"{self._render_type_ref(gql_type.of_type)}!"
        if isinstance(gql_type, GraphQLList):
            return f"[{self._render_type_ref(gql_type.of_type)}]"
        return gql_type.name

    def _is_object_type(self, gql_type: Any) -> bool:
        """``gql_type`` 이 :class:`GraphQLObjectType` 이면 True 를 반환한다."""
        return isinstance(gql_type, GraphQLObjectType)

    def _is_excluded_type(self, name: str) -> bool:
        return name in self.excluded_type_names or name.startswith(
            self.excluded_type_prefixes,
        )

    def _is_excluded_field(self, name: str) -> bool:
        return name.startswith(self.excluded_field_prefixes)

    def _fragment_name(self, type_name: str) -> str:
        return f"{type_name}{self.fragment_suffix}"

    def _reuse_hint(self, type_name: str) -> str:
        """Cycle 를 끊은 selection 위에 재사용 가능한 fragment 를 표시하는 comment.

        Fragment cycle (GraphQL spec § 5.5.2.2 *No Fragment Cycles*) 을 끊는
        최소 fallback 바로 위에 ``# ...XxxFragment`` 형태로 출력한다. 여기서
        직접 spread 하면 부정한 fragment cycle 이 되지만, comment 덕분에 이
        reference field 를 어떤 기존 fragment 가 cover 하는지 생성 파일에서
        한눈에 알 수 있다.
        """
        return f"# ...{self._fragment_name(type_name)}"

    def _cycle_break_selection(self, obj_type: GraphQLObjectType) -> str:
        """Cycle 를 끊은 self-relation field 에 출력하는 최소 selection.

        Cycle 분기의 reuse hint (:meth:`_reuse_hint`) 아래에서, 또는 self-FK
        분기 (``prev { id }`` 처럼 hint 없이) 에서 쓰인다. 두 경우 모두 fragment
        를 spread 하면 cycle 이 되므로, body 는 모든 scalar 대신 ``id`` 하나로
        축약한다 — selection set 을 valid 하게 유지하고 object 를 식별하기에 딱
        충분한 만큼이다. Type 이 ``id`` 를 노출하면 (모든 ``relay.Node`` 기반
        ``DjangoObjectType`` 이 그렇다) ``id`` 를, 아니면 selection set 이 비지
        않도록 :attr:`cycle_break_fallback` (``__typename``) 으로 fallback 한다.
        """
        if "id" in obj_type.fields:
            return "id"
        return self.cycle_break_fallback

    def _render_default_value(self, value: Any) -> str | None:
        """Argument 의 default value 를 GraphQL literal 로 렌더링한다.

        Default 를 출력하지 않아야 할 때 (``Undefined``, ``None``, 또는
        지원하지 않는 복합 type) 는 ``None`` 을 반환한다.
        """
        if value is Undefined or value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return None

    # ──────────────────────────────────────────────────────────────────
    # Fragment collection (BFS with cycle protection)
    # ──────────────────────────────────────────────────────────────────

    def _collect_object_types(self) -> dict[str, GraphQLObjectType]:
        """Root operation 을 walk 하여 도달 가능한 모든 ObjectType 을 수집한다."""
        visited: dict[str, GraphQLObjectType] = {}
        queue: deque[GraphQLObjectType] = deque()

        roots = (
            self.schema.query_type,
            self.schema.mutation_type,
            self.schema.subscription_type,
        )
        for root in roots:
            if root is None:
                continue
            for fld in root.fields.values():
                inner = self._unwrap(fld.type)
                if self._is_object_type(inner):
                    queue.append(inner)

        while queue:
            obj = queue.popleft()
            if obj.name in visited or self._is_excluded_type(obj.name):
                continue
            visited[obj.name] = obj
            for fname, fld in obj.fields.items():
                if self._is_excluded_field(fname):
                    continue
                inner = self._unwrap(fld.type)
                if (
                    self._is_object_type(inner)
                    and inner.name not in visited
                    and not self._is_excluded_type(inner.name)
                ):
                    queue.append(inner)

        return visited

    def _compute_reachability(
        self,
        types: dict[str, GraphQLObjectType],
    ) -> dict[str, frozenset[str]]:
        """각 type 마다, 그 type 에서 transitive 하게 도달 가능한 type 이름 집합.

        :meth:`_render_fragment` 가 cycle (GraphQL spec § 5.5.2.2 *No Fragment
        Cycles*) 을 탐지하는 데 사용한다.
        """
        reach: dict[str, frozenset[str]] = {}
        for name, root in types.items():
            seen: set[str] = set()
            stack: list[GraphQLObjectType] = [root]
            while stack:
                cur = stack.pop()
                for fname, fld in cur.fields.items():
                    if self._is_excluded_field(fname):
                        continue
                    inner = self._unwrap(fld.type)
                    if not self._is_object_type(inner) or self._is_excluded_type(
                        inner.name,
                    ):
                        continue
                    if inner.name in seen:
                        continue
                    seen.add(inner.name)
                    nxt = types.get(inner.name)
                    if nxt is not None:
                        stack.append(nxt)
            reach[name] = frozenset(seen)
        return reach

    # ──────────────────────────────────────────────────────────────────
    # Owner resolution
    # ──────────────────────────────────────────────────────────────────

    def _owner_label(self, gql_type: GraphQLObjectType) -> str | None:  # noqa: C901
        """``gql_type`` 을 소유한 registry app label 을, 없으면 ``None`` 을 반환한다.

        판별 순서:

        1. ``DjangoObjectType``: ``_meta.model._meta.app_label`` 을 쓴다.
        2. 순수 graphene ObjectType: 가장 길게 매칭되는 ``module_prefix``.
        3. 자동 생성된 Connection / Edge type: 내부 ``node`` field 의 type 에
           위임한다.
        """
        graphene_type = getattr(gql_type, "graphene_type", None)
        if graphene_type is not None:
            meta = getattr(graphene_type, "_meta", None)
            model = getattr(meta, "model", None) if meta is not None else None
            if model is not None:
                return model._meta.app_label  # noqa: SLF001

            module = getattr(graphene_type, "__module__", "") or ""
            best_label: str | None = None
            best_len = -1
            for label, spec in self.registry.items():
                prefix = spec.module_prefix
                if (module == prefix or module.startswith(prefix + ".")) and len(
                    prefix,
                ) > best_len:
                    best_label = label
                    best_len = len(prefix)
            if best_label is not None:
                return best_label

        fields = getattr(gql_type, "fields", None)
        if fields:
            # Edge type: delegate to ``node`` field's inner type.
            node_field = fields.get("node")
            if node_field is not None:
                inner = self._unwrap(node_field.type)
                if isinstance(inner, GraphQLObjectType) and inner is not gql_type:
                    return self._owner_label(inner)
            # Connection type: recurse via ``edges`` → Edge → ``node``.
            edges_field = fields.get("edges")
            if edges_field is not None:
                edge_type = self._unwrap(edges_field.type)
                if (
                    isinstance(edge_type, GraphQLObjectType)
                    and edge_type is not gql_type
                ):
                    return self._owner_label(edge_type)

        return None

    def _group_types_by_owner(
        self,
        types: dict[str, GraphQLObjectType],
    ) -> dict[str, list[GraphQLObjectType]]:
        """``types`` 를 소유 app label 별로 분할한다 (불명 → :data:`SHARED_LABEL`)."""
        grouped: dict[str, list[GraphQLObjectType]] = defaultdict(list)
        for name in sorted(types):
            label = self._owner_label(types[name]) or SHARED_LABEL
            grouped[label].append(types[name])
        return dict(grouped)

    # ──────────────────────────────────────────────────────────────────
    # App-local operation field discovery
    # ──────────────────────────────────────────────────────────────────

    def _collect_app_operation_field_names(
        self,
        spec: AppSpec,
    ) -> dict[str, set[str]]:
        """Schema module 에서 operation kind 별 field 이름 집합을 수집한다."""
        out: dict[str, set[str]] = {
            "Query": set(),
            "Mutation": set(),
            "Subscription": set(),
        }
        for module_name in spec.iter_schema_modules():
            try:
                mod = importlib.import_module(module_name)
            except ImportError:
                continue
            for kind, names in out.items():
                cls = getattr(mod, kind, None)
                if cls is None:
                    continue
                meta = getattr(cls, "_meta", None)
                fields = getattr(meta, "fields", None) if meta is not None else None
                if not fields:
                    continue
                # graphene declares fields in snake_case; the root GraphQL schema
                # exposes them in camelCase (default ``auto_camelcase=True``).
                names.update(to_camel_case(n) for n in fields)
        return out

    # ──────────────────────────────────────────────────────────────────
    # Fragment rendering
    # ──────────────────────────────────────────────────────────────────

    def _scalar_only_fields(self, obj_type: GraphQLObjectType) -> list[str]:
        """Unwrap 한 type 이 ObjectType 이 아닌 ``obj_type`` field 이름을 반환한다."""
        names: list[str] = []
        for fname, fld in obj_type.fields.items():
            if self._is_excluded_field(fname):
                continue
            inner = self._unwrap(fld.type)
            if not self._is_object_type(inner):
                names.append(fname)
        return names

    def _is_relay_edge_type(self, obj_type: GraphQLObjectType) -> bool:
        """구조적 형태로 Relay 자동 생성 Edge ObjectType 을 탐지한다.

        ``obj_type`` 의 이름이 ``...Edge`` 이고 ``node`` 와 ``cursor`` field 를
        모두 노출하면 True — 이는 ``graphene.relay.Connection`` 이 node type
        주위에 합성하는 Edge wrapper 의 구조적 지문이다. Edge type 은 domain
        model 이 아니라 framework wrapper 이므로 standalone fragment 로는 결코
        출력하지 않는다 — 대신 Connection body 가 Edge body
        (``edges { node { ... } cursor }``) 를 inline 한다.

        Args:
            obj_type: 검사할 GraphQLObjectType.

        Returns:
            ``obj_type`` 이 Relay Edge 형태와 일치할 때만 True.
        """
        if not obj_type.name.endswith("Edge"):
            return False
        field_names = set(obj_type.fields.keys())
        return {"node", "cursor"} <= field_names

    def _is_relay_connection_type(self, obj_type: GraphQLObjectType) -> bool:
        """구조적 형태로 Relay 자동 생성 Connection ObjectType 을 탐지한다.

        ``obj_type`` 의 이름이 ``...Connection`` 이고 ``pageInfo`` 와 ``edges``
        field 를 모두 노출하면 True — 이는 ``graphene.relay.Connection`` 이 node
        type 주위에 합성하는 wrapper 의 구조적 지문이다. Connection type 은
        domain model 이 아니라 framework wrapper 이므로 standalone fragment 로는
        결코 출력하지 않는다 — 대신 parent fragment 가 Connection body
        (``pageInfo { ... } edges { node { ... } cursor } ...``) 를 inline 하고,
        내부 ``node`` 는 전용 Django model ``...XxxTypeFragment`` 정의를 spread
        한다.

        Args:
            obj_type: 검사할 GraphQLObjectType.

        Returns:
            ``obj_type`` 이 Relay Connection 형태와 일치할 때만 True.
        """
        if not obj_type.name.endswith("Connection"):
            return False
        field_names = set(obj_type.fields.keys())
        return {"pageInfo", "edges"} <= field_names

    def _backing_model(self, obj_type: GraphQLObjectType) -> type | None:
        """``obj_type`` 의 backing Django model, 없으면 None.

        ``graphene_type._meta.model`` 을 읽는다 — ``DjangoObjectType`` subclass
        에는 채워져 있으나 framework wrapper (Relay ``Connection`` / ``Edge``,
        Mutation payload, …) 에는 없다.
        """
        graphene_type = getattr(obj_type, "graphene_type", None)
        if graphene_type is None:
            return None
        meta = getattr(graphene_type, "_meta", None)
        if meta is None:
            return None
        return getattr(meta, "model", None)

    def _model_relation_field_names(
        self,
        obj_type: GraphQLObjectType,
    ) -> frozenset[str]:
        """관계형 Django model field 의 camelCase GraphQL field 이름을 반환한다.

        ``obj_type`` 뒤의 Django model 을 (있으면) 검사하여 ``ForeignKey``,
        ``ManyToManyField``, ``OneToOneField`` 또는 그 reverse 에 해당하는
        GraphQL field 이름 집합을 — schema 기본값 ``auto_camelcase=True`` 에
        맞춰 camelCase 로 — 반환한다.

        :meth:`_render_inline_selection` 은 cycle 를 끊은 inline body 에서
        관계형 field 를 omit 하는 데 (전용 ``...XxxFragment`` 정의가 이미
        노출한 내용을 중복하므로), :meth:`_render_fragment` 는 self-FK
        (``fname { id }``) 판정과 비-self 관계형 field 의 fragment-spread
        판정에 이 집합을 사용한다.

        ``obj_type`` 이 Django model 로 backing 되지 않으면 (예: Relay 생성
        ``Connection`` / ``Edge`` type 이나 graphene 전용 ObjectType wrapper)
        빈 :class:`frozenset` 을 반환한다.

        Args:
            obj_type: introspection 할 GraphQLObjectType.

        Returns:
            camelCase field 이름의 frozenset. Django model 이 없으면 빈 집합.
        """
        model = self._backing_model(obj_type)
        if model is None:
            return frozenset()
        names: set[str] = set()
        for field in model._meta.get_fields():  # noqa: SLF001
            if not field.is_relation:
                continue
            accessor = getattr(field, "get_accessor_name", None)
            if callable(accessor):
                accessor_name = accessor()
                if accessor_name:
                    names.add(to_camel_case(accessor_name))
                    continue
            names.add(to_camel_case(field.name))
        return frozenset(names)

    def _render_inline_selection(
        self,
        obj_type: GraphQLObjectType,
        path: frozenset[str],
        outer_name: str,
        reach: dict[str, frozenset[str]],
    ) -> list[str]:
        """``obj_type`` 을 안전하게 inline 한 indent 없는 selection-set 줄을 반환한다.

        ``obj_type`` 자체의 fragment-spread 가 불가능할 때 쓴다 — ``obj_type``
        이 standalone fragment 로 일부러 출력하지 않는 Relay framework wrapper
        (``Connection`` / ``Edge``) 이거나, spread 하면 *outer* fragment 에
        대해 GraphQL spec § 5.5.2.2 를 위반하기 때문이다.

        Field 별 selection 규칙:

        1. Field 이름이 ``obj_type`` 의 backing model 에서 Django 관계형 field
           (``ForeignKey``, ``ManyToManyField``, ``OneToOneField`` 또는 그
           reverse) 에 해당하면 그 field 는 *omit* 한다. 이 field 들은 outer
           Django model fragment 의 전용 ``...XxxFragment`` 정의가 이미
           노출한다.
        2. ``inner`` 이 Django model 로 backing 되고 그 이름이 outer fragment
           (``outer_name``) 와 현재 ``obj_type`` 둘 다와 다르면
           ``...InnerFragment`` 를 출력한다 — Django model fragment 는
           self-contained 라 fragment-spread graph 가 acyclic 하게 유지된다.
        3. ``inner`` 이 Django model 로 backing 되지만 ``outer_name`` 또는
           ``obj_type.name`` 과 같으면 spread 가 fragment cycle 을 만든다 —
           ``# ...InnerFragment`` reuse hint (:meth:`_reuse_hint`) 와 그 아래
           ``id`` (:meth:`_cycle_break_selection`) 를 출력한다.
        4. ``inner`` 이 Relay ``Connection`` / ``Edge`` framework wrapper 이면
           inline 전개로 재귀한다 (type 이 이미 ``path`` 에 있으면 scalar
           fallback).
        5. 그 외 (다른 framework ObjectType, 예: ``PageInfo``): schema 가 outer
           fragment 로 되돌아오지 않으면 spread 하고, 되돌아오면 inline 재귀
           또는 scalar fallback 한다.

        종료는 ``path`` 의 단조 증가와 schema 내 ObjectType 의 유한 개수로
        보장된다: 모든 재귀는 skip 하거나, collapse 하거나, ``path`` 에 새 type
        이름을 추가한다. Depth cap 은 필요 없다.

        Args:
            obj_type: inline 할 ObjectType.
            path: inline chain 에 이미 있는 조상 type 이름들.
            outer_name: fragment 의 root ObjectType 이름.
            reach: ``_compute_reachability`` 가 만든 transitive reachability map.

        Returns:
            앞쪽 indent 가 없는 selection 줄 목록.
        """
        lines: list[str] = []
        relation_names = self._model_relation_field_names(obj_type)
        for fname, fld in obj_type.fields.items():
            if self._is_excluded_field(fname):
                continue
            inner = self._unwrap(fld.type)
            if not (
                self._is_object_type(inner) and not self._is_excluded_type(inner.name)
            ):
                lines.append(fname)
                continue
            if fname in relation_names:
                continue
            lines.append(f"{fname} {{")
            backing = self._backing_model(inner)
            is_wrapper = self._is_relay_connection_type(
                inner,
            ) or self._is_relay_edge_type(inner)
            if backing is not None:
                if inner.name == outer_name or inner.name == obj_type.name:
                    lines.append(f"{self.indent}{self._reuse_hint(inner.name)}")
                    lines.append(
                        f"{self.indent}{self._cycle_break_selection(inner)}",
                    )
                else:
                    lines.append(
                        f"{self.indent}...{self._fragment_name(inner.name)}",
                    )
            elif is_wrapper:
                if inner.name in path:
                    scalars = self._scalar_only_fields(inner) or [
                        self.cycle_break_fallback,
                    ]
                    for s in scalars:
                        lines.append(f"{self.indent}{s}")
                else:
                    sub = self._render_inline_selection(
                        inner,
                        path | {obj_type.name},
                        outer_name,
                        reach,
                    )
                    lines.extend(f"{self.indent}{ln}" for ln in sub)
            else:
                inner_reach = reach.get(inner.name, frozenset())
                cycles_to_outer = outer_name in inner_reach
                if not cycles_to_outer:
                    lines.append(
                        f"{self.indent}...{self._fragment_name(inner.name)}",
                    )
                elif inner.name in path:
                    scalars = self._scalar_only_fields(inner) or [
                        self.cycle_break_fallback,
                    ]
                    for s in scalars:
                        lines.append(f"{self.indent}{s}")
                else:
                    sub = self._render_inline_selection(
                        inner,
                        path | {obj_type.name},
                        outer_name,
                        reach,
                    )
                    lines.extend(f"{self.indent}{ln}" for ln in sub)
            lines.append("}")
        if not lines:
            lines.append(self.cycle_break_fallback)
        return lines

    def _render_fragment(
        self,
        obj_type: GraphQLObjectType,
        reach: dict[str, frozenset[str]],
    ) -> str:
        """단일 ObjectType 을 GraphQL fragment 정의로 렌더링한다.

        하위 object 처리 우선순위 (검사 순서대로):

        1. ``fname`` 이 self-referential FK (prev / next / parent 처럼
           ``obj_type`` 의 backing model 에서 같은 model 을 가리키는 관계형
           field, ``inner.name == obj_type.name``) 이면 ``fname { id }`` 만
           출력한다. 연결된 row 의 식별자면 충분하고, 여기서 fragment 를 spread
           하면 어차피 cycle (GraphQL spec § 5.5.2.2) 이 된다.
        2. ``inner`` 이 Relay 자동 생성 ``Connection`` 또는 ``Edge`` type
           (``_is_relay_connection_type`` / ``_is_relay_edge_type``) 이면
           :meth:`_render_inline_selection` 으로 그 body 를 inline 하여 outer
           fragment 가 wrapper 구조 (``pageInfo { ... } edges { node {
           ...XxxTypeFragment } cursor } totalCount ...``) 를 직접 노출하게
           한다. Wrapper fragment 는 standalone 정의로 출력하지 않으며, 내부
           ``node`` 는 전용 Django model fragment 로 fallback 한다.
        3. ``fname`` 이 Django 관계형 field (FK / M2M / OneToOne forward+reverse)
           이고 ``inner.name != obj_type.name`` 이면 fragment spread
           ``...{InnerFragment}`` 를 출력한다. 생성된 fragment 는
           self-contained 다 — ``_render_inline_selection`` 이 cycle 를 끊은
           inline 전개 중 model 관계 field 를 omit 하므로, schema graph 에
           cycle 이 있어도 fragment-spread graph 는 acyclic 하게 유지된다
           (GraphQL spec § 5.5.2.2).
        4. 그 외: ``...InnerFragment`` spread 가 schema cycle
           (``inner.name == obj_type.name`` 또는 ``obj_type.name in
           reach[inner.name]``) 을 만들면 ``# ...InnerFragment`` reuse hint
           (:meth:`_reuse_hint`) 와 그 아래 ``id``
           (:meth:`_cycle_break_selection`) 를 출력하고, 아니면 spread 한다.

        Framework 생성 ObjectType (Relay ``Connection`` / ``Edge``, Mutation
        payload, …) 은 Django model 이 없어 ``relation_names`` 가 비므로,
        기존 cycle 판정이 그대로 적용된다.

        Args:
            obj_type: 렌더링할 ObjectType.
            reach: :meth:`_compute_reachability` 가 만든 transitive
                reachability map.

        Returns:
            개행 두 개로 끝나는 단일 문자열 형태의 렌더링된 fragment 정의.
        """
        relation_names = self._model_relation_field_names(obj_type)
        body_lines: list[str] = []
        for fname, fld in obj_type.fields.items():
            if self._is_excluded_field(fname):
                continue
            inner = self._unwrap(fld.type)
            if self._is_object_type(inner) and not self._is_excluded_type(inner.name):
                # Self-referential FK (prev / next / parent → same model): emit
                # only ``id`` underneath. The linked row's identity is enough;
                # spreading the fragment here would form a cycle (GraphQL spec
                # § 5.5.2.2) and restating every scalar adds nothing.
                if fname in relation_names and inner.name == obj_type.name:
                    body_lines.append(f"{fname} {{")
                    body_lines.append(
                        f"{self.indent}{self._cycle_break_selection(inner)}",
                    )
                    body_lines.append("}")
                    continue
                body_lines.append(f"{fname} {{")
                if self._is_relay_connection_type(
                    inner,
                ) or self._is_relay_edge_type(inner):
                    inline = self._render_inline_selection(
                        inner,
                        path=frozenset({obj_type.name}),
                        outer_name=obj_type.name,
                        reach=reach,
                    )
                    for ln in inline:
                        body_lines.append(f"{self.indent}{ln}")
                elif fname in relation_names and inner.name != obj_type.name:
                    body_lines.append(
                        f"{self.indent}...{self._fragment_name(inner.name)}",
                    )
                else:
                    inner_reach = reach.get(inner.name, frozenset())
                    cycles_back = (
                        inner.name == obj_type.name or obj_type.name in inner_reach
                    )
                    if cycles_back:
                        body_lines.append(
                            f"{self.indent}{self._reuse_hint(inner.name)}",
                        )
                        body_lines.append(
                            f"{self.indent}{self._cycle_break_selection(inner)}",
                        )
                    else:
                        body_lines.append(
                            f"{self.indent}...{self._fragment_name(inner.name)}",
                        )
                body_lines.append("}")
            else:
                body_lines.append(fname)

        body = "\n".join(f"{self.indent}{ln}" for ln in body_lines)
        return (
            f"fragment {self._fragment_name(obj_type.name)} "
            f"on {obj_type.name} {{\n{body}\n}}\n\n"
        )

    # ──────────────────────────────────────────────────────────────────
    # Operation rendering
    # ──────────────────────────────────────────────────────────────────

    def _render_arguments(
        self,
        args: dict[str, GraphQLArgument],
    ) -> tuple[list[str], list[str]]:
        """``($var: Type)`` 선언과 ``name: $var`` pass-through 를 만든다."""
        variable_decls: list[str] = []
        argument_passes: list[str] = []
        for arg_name, arg in args.items():
            type_ref = self._render_type_ref(arg.type)
            decl = f"${arg_name}: {type_ref}"
            default = self._render_default_value(arg.default_value)
            if default is not None:
                decl += f" = {default}"
            variable_decls.append(decl)
            argument_passes.append(f"{arg_name}: ${arg_name}")
        return variable_decls, argument_passes

    def _render_operation(
        self,
        kind: str,
        name: str,
        field: GraphQLField,
        reach: dict[str, frozenset[str]],
    ) -> str:
        """단일 root field 를 ``query``/``mutation``/``subscription`` 으로 렌더링한다.

        Root field 가 Relay framework wrapper (``Connection`` / ``Edge``) 를
        반환하면 — standalone fragment 로는 출력하지 않으므로 —
        :meth:`_render_inline_selection` 으로 body 를 inline 한다. 내부 ``node``
        (Django model) 는 전용 ``...XxxTypeFragment`` spread 로 fallback 한다.
        그 외 모든 ObjectType 반환 type 은 body 가 단일 ``...InnerFragment``
        spread 이다.

        Args:
            kind: ``"query"`` / ``"mutation"`` / ``"subscription"``.
            name: root field 이름.
            field: root field 의 GraphQLField.
            reach: :meth:`_compute_reachability` 가 만든 transitive
                reachability map.

        Returns:
            개행 두 개로 끝나는 단일 문자열 형태의 렌더링된 operation 정의.
        """
        variable_decls, argument_passes = self._render_arguments(field.args)
        var_str = ", ".join(variable_decls)
        arg_str = ", ".join(argument_passes)

        head = f"{kind} {name}"
        if var_str:
            head += f"({var_str})"

        call = f"{name}({arg_str})" if arg_str else name
        inner = self._unwrap(field.type)
        if self._is_object_type(inner) and not self._is_excluded_type(inner.name):
            if self._is_relay_connection_type(
                inner,
            ) or self._is_relay_edge_type(inner):
                inline = self._render_inline_selection(
                    inner,
                    path=frozenset({inner.name}),
                    outer_name="",
                    reach=reach,
                )
                body_lines = [f"{self.indent}{call} {{"]
                body_lines.extend(f"{self.indent * 2}{ln}" for ln in inline)
                body_lines.append(f"{self.indent}}}")
                body = "\n".join(body_lines)
            else:
                body = (
                    f"{self.indent}{call} {{\n"
                    f"{self.indent * 2}...{self._fragment_name(inner.name)}\n"
                    f"{self.indent}}}"
                )
        else:
            body = f"{self.indent}{call}"

        return f"{head} {{\n{body}\n}}\n\n"

    def _iter_app_operations(
        self,
        kind: str,
        root: GraphQLObjectType | None,
        field_names: set[str],
        reach: dict[str, frozenset[str]],
    ) -> Iterator[str]:
        """``field_names`` 에 속한 field 들의 렌더링된 operation 을 yield 한다."""
        if root is None or not field_names:
            return
        for name, fld in root.fields.items():
            if name not in field_names or self._is_excluded_field(name):
                continue
            yield self._render_operation(kind, name, fld, reach)

    # ──────────────────────────────────────────────────────────────────
    # File output
    # ──────────────────────────────────────────────────────────────────

    def _write(self, path: Path, source: Iterable[str]) -> int:
        """Chunk 들을 ``path`` 에 쓰고, 쓴 chunk 개수를 반환한다.

        각 chunk 는 렌더링된 fragment 또는 operation 하나에 대응하므로, 반환값은
        해당 file 의 GraphQL 정의 개수와 같다.
        """
        count = 0
        with open(path, "w", encoding="utf-8") as fp:
            for chunk in source:
                fp.write(chunk)
                count += 1
        return count

    def _iter_all_operations(
        self,
        kind: str,
        root: GraphQLObjectType | None,
        reach: dict[str, frozenset[str]],
    ) -> Iterator[str]:
        """``root`` 의 제외되지 않은 모든 field 의 렌더링된 operation 을 yield 한다."""
        if root is None:
            return
        for name, fld in root.fields.items():
            if self._is_excluded_field(name):
                continue
            yield self._render_operation(kind, name, fld, reach)

    @classmethod
    def codegen(
        cls,
        schema: Any,
        graphqls: str | Path,
        **kwargs: Any,
    ) -> tuple[GQLGenerator, dict[str, dict[str, int]]]:
        """App 별 ``.graphql`` file 과 legacy aggregate file 을 생성한다.

        출력은 세 계층이다:

        1. **Aggregate** (legacy): 출력 root 의 ``fragments.graphql``,
           ``queries.graphql``, ``mutations.graphql``, ``subscriptions.graphql``
           — 도달 가능한 모든 fragment 와 모든 root operation.
        2. **Shared**: 출력 root 의 :data:`GLOBAL_FILENAME` — 등록된 app 에
           귀속할 수 없는 ObjectType 의 fragment.
        3. **Per-app**: ``<app>/{fragments,queries,mutations,subscriptions}.graphql``
           — 해당 app 이 소유한 fragment 와 app 의 local schema module 에 선언된
           operation.

        Args:
            schema: ``graphene.Schema`` 또는 ``graphql.GraphQLSchema``.
            graphqls: 출력 directory 경로.
            **kwargs: :meth:`__init__` 으로 전달.

        Returns:
            (generator, summary) tuple. ``summary`` 는 bucket label
            (:data:`AGGREGATE_LABEL`, :data:`SHARED_LABEL`, 또는 app label) 을
            ``{filename: chunk_count}`` 에 매핑한다.
        """
        instance = cls(schema, **kwargs)
        out_dir = Path(graphqls)
        out_dir.mkdir(parents=True, exist_ok=True)

        types = instance._collect_object_types()
        reach = instance._compute_reachability(types)
        grouped = instance._group_types_by_owner(types)

        roots: dict[str, GraphQLObjectType | None] = {
            "query": instance.schema.query_type,
            "mutation": instance.schema.mutation_type,
            "subscription": instance.schema.subscription_type,
        }

        summary: dict[str, dict[str, int]] = defaultdict(dict)

        # 1. Aggregate files at output root (legacy behavior).
        sorted_types = [
            types[n]
            for n in sorted(types)
            if not instance._is_relay_edge_type(types[n])
            and not instance._is_relay_connection_type(types[n])
        ]
        summary[AGGREGATE_LABEL][cls.FRAGMENT_FILENAME] = instance._write(
            out_dir / cls.FRAGMENT_FILENAME,
            (instance._render_fragment(t, reach) for t in sorted_types),
        )
        for kind, filename in cls.OPERATION_FILENAMES.items():
            summary[AGGREGATE_LABEL][filename] = instance._write(
                out_dir / filename,
                instance._iter_all_operations(kind, roots[kind], reach),
            )

        # 2. Shared fragments → global.graphql at output root.
        # 3. Per-app fragments under <app>/fragments.graphql.
        for label, app_types in grouped.items():
            if not app_types:
                continue
            if label == SHARED_LABEL:
                summary[SHARED_LABEL][GLOBAL_FILENAME] = instance._write(
                    out_dir / GLOBAL_FILENAME,
                    (
                        instance._render_fragment(t, reach)
                        for t in app_types
                        if not instance._is_relay_edge_type(t)
                        and not instance._is_relay_connection_type(t)
                    ),
                )
                continue
            app_dir = out_dir / label
            app_dir.mkdir(parents=True, exist_ok=True)
            summary[label][cls.FRAGMENT_FILENAME] = instance._write(
                app_dir / cls.FRAGMENT_FILENAME,
                (
                    instance._render_fragment(t, reach)
                    for t in app_types
                    if not instance._is_relay_edge_type(t)
                    and not instance._is_relay_connection_type(t)
                ),
            )

        # 4. Per-app operations.
        for label, spec in instance.registry.items():
            field_names = instance._collect_app_operation_field_names(spec)
            app_dir = out_dir / label
            for kind, filename in cls.OPERATION_FILENAMES.items():
                names = field_names.get(kind.capitalize(), set())
                ops = list(
                    instance._iter_app_operations(kind, roots[kind], names, reach),
                )
                if not ops:
                    continue
                app_dir.mkdir(parents=True, exist_ok=True)
                summary[label][filename] = instance._write(
                    app_dir / filename,
                    ops,
                )

        return instance, dict(summary)


class Command(BaseCommand):
    """Root schema 로부터 Django app 별 ``.graphql`` file 을 생성한다.

    프로젝트의 root schema (``project.schema.schema``) 를 walk 하여, 등록된 각
    Django app 마다 최대 네 개의 file 을 담은 하위 directory 를 작성한다:

    - ``fragments.graphql`` — app 이 소유한 ObjectType 마다 fragment 하나
    - ``queries.graphql`` — app 이 선언한 모든 ``Query`` field
    - ``mutations.graphql`` — app 이 선언한 모든 ``Mutation`` field
    - ``subscriptions.graphql`` — app 이 선언한 모든 ``Subscription`` field

    빈 file 은 건너뛴다. 어느 등록 app 에도 귀속할 수 없는 type 은 ``_shared/``
    bucket 에 들어간다.
    """

    help = (
        "Generate per-Django-app GraphQL fragments / queries / mutations / "
        "subscriptions files from the root schema into the given directory "
        "(default: graphqls)."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """CLI argument 를 등록한다.

        Args:
            parser: Django 가 제공하는 argparse parser.
        """
        parser.add_argument(
            "graphqls",
            nargs="?",
            default="graphqls",
            help=(
                "Output directory for the generated per-app .graphql files. "
                "Relative paths are resolved against settings.BASE_DIR. "
                "Defaults to 'graphqls'."
            ),
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Codegen pipeline 을 실행한다.

        Args:
            *args: 사용하지 않는 positional arg.
            **options: 파싱된 CLI option. ``graphqls`` (str) 를 기대한다.
        """
        # Lazy import — avoids loading the full schema at module import time
        # (so e.g. ``manage.py help`` does not pay the cost).
        from project.schema import schema

        out_dir = Path(str(options["graphqls"]))
        if not out_dir.is_absolute():
            out_dir = Path(settings.BASE_DIR) / out_dir

        self.stdout.write(f"Generating GraphQL operations into: {out_dir}")
        _, summary = GQLGenerator.codegen(schema, out_dir)

        if not summary:
            self.stdout.write(self.style.WARNING("No outputs were produced."))
            self.stdout.write(self.style.SUCCESS("Done."))
            return

        aggregate = summary.get(AGGREGATE_LABEL, {})
        shared = summary.get(SHARED_LABEL, {})
        app_labels = sorted(
            k for k in summary if k not in {AGGREGATE_LABEL, SHARED_LABEL}
        )

        # Per-app outputs.
        for label in app_labels:
            self.stdout.write(f"  {label}/")
            for filename in sorted(summary[label]):
                count = summary[label][filename]
                self.stdout.write(f"    {filename:>22s}  {count:>5d} items")

        # Missing breakdown: aggregate - (per-app + shared).
        # Aggregate counts are computed against the merged root schema and
        # are not printed on their own — they only surface here so callers
        # can verify that every fragment / operation was attributed.
        # Maps each aggregate filename → counterpart per-app filename.
        aggregate_to_app_filename: dict[str, str] = {
            GQLGenerator.FRAGMENT_FILENAME: GQLGenerator.FRAGMENT_FILENAME,
            **{fn: fn for fn in GQLGenerator.OPERATION_FILENAMES.values()},
        }
        self.stdout.write("  Results:")
        for agg_filename, app_filename in aggregate_to_app_filename.items():
            total = aggregate.get(agg_filename, 0)
            covered = sum(summary[lbl].get(app_filename, 0) for lbl in app_labels)
            if agg_filename == GQLGenerator.FRAGMENT_FILENAME:
                covered += shared.get(GLOBAL_FILENAME, 0)
            missing = total - covered
            label_style = self.style.WARNING if missing else self.style.SUCCESS
            self.stdout.write(
                label_style(
                    f"    {agg_filename:>22s}  {missing:>5d} / {total:>5d}",
                ),
            )
        # ``global.graphql`` is not part of the aggregate vs per-app math —
        # show its raw item count separately.
        for shared_filename in sorted(shared):
            shared_count = shared[shared_filename]
            self.stdout.write(
                f"    {shared_filename:>22s}  {shared_count:>5d} items",
            )

        self.stdout.write(self.style.SUCCESS("graphql_codegen done."))
