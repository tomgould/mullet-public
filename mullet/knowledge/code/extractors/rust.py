"""
Rust code extractor using tree-sitter.

Extracts functions, methods, structs, enums, traits, impl blocks,
constants, modules, macros, and type aliases.
"""

import logging
from typing import Optional

from .base import BaseExtractor, CodeChunk

logger = logging.getLogger(__name__)


class RustExtractor(BaseExtractor):
    """Extract code chunks from Rust source files."""

    language = "rust"
    tree_sitter_language = "rust"

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract symbols from Rust code."""
        chunks: list[CodeChunk] = []
        source_bytes = bytes(source_code, "utf-8")

        try:
            tree = self.parse(source_code)
        except Exception as e:
            logger.warning(f"Failed to parse Rust file {file_path}: {e}")
            return chunks

        root = tree.root_node

        for node in root.children:
            if node.type == "function_item":
                chunk = self._extract_function(node, source_bytes, file_path)
                if chunk:
                    chunks.append(chunk)
            elif node.type == "struct_item":
                chunk = self._extract_struct(node, source_bytes, file_path)
                if chunk:
                    chunks.append(chunk)
            elif node.type == "enum_item":
                chunk = self._extract_enum(node, source_bytes, file_path)
                if chunk:
                    chunks.append(chunk)
            elif node.type == "trait_item":
                chunk = self._extract_trait(node, source_bytes, file_path)
                if chunk:
                    chunks.append(chunk)
            elif node.type == "impl_item":
                chunks.extend(self._extract_impl(node, source_bytes, file_path))
            elif node.type in ("const_item", "static_item"):
                chunk = self._extract_constant(node, source_bytes, file_path)
                if chunk:
                    chunks.append(chunk)
            elif node.type == "type_item":
                chunk = self._extract_type_alias(node, source_bytes, file_path)
                if chunk:
                    chunks.append(chunk)
            elif node.type == "mod_item":
                chunks.extend(self._extract_mod(node, source_bytes, file_path))
            elif node.type == "macro_definition":
                chunk = self._extract_macro(node, source_bytes, file_path)
                if chunk:
                    chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_name(self, node, source_bytes: bytes) -> Optional[str]:
        """Get the identifier / type_identifier name for a node."""
        for child in node.children:
            if child.type in ("identifier", "type_identifier"):
                return self.get_node_text(child, source_bytes)
        return None

    def _get_signature(self, node, source_bytes: bytes) -> str:
        """Build a signature like (args) -> Ret from a function-ish node."""
        params = self.find_child_by_type(node, "parameters")
        ret = self.find_child_by_type(node, "return_type")
        sig = self.get_node_text(params, source_bytes) if params else "()"
        if ret:
            sig = f"{sig} -> {self.get_node_text(ret, source_bytes)}"
        return sig

    def _extract_docstring(self, node, source_bytes: bytes) -> Optional[str]:
        """Collect consecutive /// line comments immediately above the node."""
        parent = node.parent
        if not parent:
            return None

        idx = None
        for i, child in enumerate(parent.children):
            if child == node:
                idx = i
                break
        if idx is None:
            return None

        lines: list[str] = []
        for i in range(idx - 1, -1, -1):
            sibling = parent.children[i]
            if sibling.type == "line_comment":
                text = self.get_node_text(sibling, source_bytes).strip()
                if text.startswith("///") and not text.startswith("////"):
                    # strip leading /// (and optional space)
                    doc_line = text[3:]
                    if doc_line.startswith(" "):
                        doc_line = doc_line[1:]
                    lines.append(doc_line)
                else:
                    break
            elif sibling.type not in ("\n", ";"):
                break

        if not lines:
            return None
        lines.reverse()
        return "\n".join(lines)

    def determine_visibility(self, name: str, node=None) -> str:
        """Rust visibility: pub* → public, otherwise private."""
        if node is None:
            return "private"
        vis = self.find_child_by_type(node, "visibility_modifier")
        if vis is not None:
            # Easiest path: check whether any child is "pub" or a sibling form
            for child in vis.children:
                if child.type == "pub":
                    return "public"
            # pub(crate), pub(super), pub(in path) all count as public
            if vis.children:
                return "public"
        return "private"

    def _determine_visibility_for_node(self, node, source_bytes: bytes) -> str:
        """Determine visibility for a node using source_bytes."""
        vis = self.find_child_by_type(node, "visibility_modifier")
        if vis is not None:
            vis_text = self.get_node_text(vis, source_bytes)
            if vis_text.strip().startswith("pub"):
                return "public"
        return "private"

    # ------------------------------------------------------------------
    # Extractors per symbol type
    # ------------------------------------------------------------------

    def _extract_function(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_class: Optional[str] = None,
        parent_module: Optional[str] = None,
    ) -> Optional[CodeChunk]:
        name = self._get_name(node, source_bytes)
        if not name:
            return None

        line_start, line_end = self.get_node_line_range(node)
        content = self.get_node_text(node, source_bytes)
        signature = self._get_signature(node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)
        visibility = self._determine_visibility_for_node(node, source_bytes)
        symbol_type = "method" if parent_class else "function"

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type=symbol_type,
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            signature=signature,
            docstring=docstring,
            parent_class=parent_class,
            parent_module=parent_module,
            visibility=visibility,
        )

    def _extract_struct(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_module: Optional[str] = None,
    ) -> Optional[CodeChunk]:
        name = self._get_name(node, source_bytes)
        if not name:
            return None

        line_start, line_end = self.get_node_line_range(node)
        content = self.get_node_text(node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)
        visibility = self._determine_visibility_for_node(node, source_bytes)

        fields: list[str] = []
        field_list = self.find_child_by_type(node, "field_declaration_list")
        if field_list:
            for field in self.find_children_by_type(field_list, "field_declaration"):
                field_name = self.find_child_by_type(field, "field_identifier")
                if field_name:
                    fields.append(self.get_node_text(field_name, source_bytes))

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="struct",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            parent_module=parent_module,
            visibility=visibility,
            extras={"fields": fields} if fields else {},
        )

    def _extract_enum(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_module: Optional[str] = None,
    ) -> Optional[CodeChunk]:
        name = self._get_name(node, source_bytes)
        if not name:
            return None

        line_start, line_end = self.get_node_line_range(node)
        content = self.get_node_text(node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)
        visibility = self._determine_visibility_for_node(node, source_bytes)

        variants: list[str] = []
        variant_list = self.find_child_by_type(node, "enum_variant_list")
        if variant_list:
            for variant in self.find_children_by_type(variant_list, "enum_variant"):
                variant_name = self.find_child_by_type(variant, "identifier")
                if variant_name:
                    variants.append(self.get_node_text(variant_name, source_bytes))

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="enum",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            parent_module=parent_module,
            visibility=visibility,
            extras={"variants": variants} if variants else {},
        )

    def _extract_trait(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_module: Optional[str] = None,
    ) -> Optional[CodeChunk]:
        name = self._get_name(node, source_bytes)
        if not name:
            return None

        line_start, line_end = self.get_node_line_range(node)
        content = self.get_node_text(node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)
        visibility = self._determine_visibility_for_node(node, source_bytes)

        methods: list[str] = []
        decl_list = self.find_child_by_type(node, "declaration_list")
        if decl_list:
            for child in decl_list.children:
                if child.type in ("function_signature_item", "function_item"):
                    method_name = self._get_name(child, source_bytes)
                    if method_name:
                        methods.append(method_name)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="trait",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            parent_module=parent_module,
            visibility=visibility,
            extras={"methods": methods} if methods else {},
        )

    def _extract_impl(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_module: Optional[str] = None,
    ) -> list[CodeChunk]:
        """Extract methods from an impl block."""
        chunks: list[CodeChunk] = []

        # Find the target type (the one after "for" if present)
        type_identifiers = self.find_children_by_type(node, "type_identifier")
        if not type_identifiers:
            return chunks

        # Check if there's a "for" keyword
        has_for = any(child.type == "for" for child in node.children)
        if has_for and len(type_identifiers) >= 2:
            impl_trait = self.get_node_text(type_identifiers[0], source_bytes)
            parent_class = self.get_node_text(type_identifiers[1], source_bytes)
            extras = {"impl_trait": impl_trait}
        else:
            parent_class = self.get_node_text(type_identifiers[0], source_bytes)
            extras = {}

        decl_list = self.find_child_by_type(node, "declaration_list")
        if decl_list:
            for child in decl_list.children:
                if child.type == "function_item":
                    chunk = self._extract_function(
                        child, source_bytes, file_path,
                        parent_class=parent_class,
                        parent_module=parent_module,
                    )
                    if chunk:
                        if extras:
                            chunk.extras.update(extras)
                        chunks.append(chunk)

        return chunks

    def _extract_constant(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_module: Optional[str] = None,
    ) -> Optional[CodeChunk]:
        name = self._get_name(node, source_bytes)
        if not name:
            return None

        line_start, line_end = self.get_node_line_range(node)
        content = self.get_node_text(node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)
        visibility = self._determine_visibility_for_node(node, source_bytes)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="constant",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            parent_module=parent_module,
            visibility=visibility,
        )

    def _extract_type_alias(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_module: Optional[str] = None,
    ) -> Optional[CodeChunk]:
        name = self._get_name(node, source_bytes)
        if not name:
            return None

        line_start, line_end = self.get_node_line_range(node)
        content = self.get_node_text(node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)
        visibility = self._determine_visibility_for_node(node, source_bytes)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="type_alias",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            parent_module=parent_module,
            visibility=visibility,
        )

    def _extract_mod(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract a module declaration or inline module."""
        chunks: list[CodeChunk] = []
        name = self._get_name(node, source_bytes)
        if not name:
            return chunks

        line_start, line_end = self.get_node_line_range(node)
        content = self.get_node_text(node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)
        visibility = self._determine_visibility_for_node(node, source_bytes)

        decl_list = self.find_child_by_type(node, "declaration_list")
        if decl_list:
            # Inline module — emit a stub for the module itself and recurse
            chunks.append(
                CodeChunk(
                    content=content,
                    source_file=file_path,
                    language=self.language,
                    symbol_type="module",
                    symbol_name=name,
                    line_start=line_start,
                    line_end=line_end,
                    docstring=docstring,
                    visibility=visibility,
                )
            )

            for child in decl_list.children:
                child_chunks = self._extract_child_with_parent_module(
                    child, source_bytes, file_path, name
                )
                chunks.extend(child_chunks)
        else:
            # External module declaration: mod foo;
            chunks.append(
                CodeChunk(
                    content=content,
                    source_file=file_path,
                    language=self.language,
                    symbol_type="module",
                    symbol_name=name,
                    line_start=line_start,
                    line_end=line_end,
                    docstring=docstring,
                    visibility=visibility,
                )
            )

        return chunks

    def _extract_child_with_parent_module(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_module: str,
    ) -> list[CodeChunk]:
        """Dispatch a child node inside a module, passing parent_module."""
        chunks: list[CodeChunk] = []

        if node.type == "function_item":
            chunk = self._extract_function(node, source_bytes, file_path, parent_module=parent_module)
            if chunk:
                chunks.append(chunk)
        elif node.type == "struct_item":
            chunk = self._extract_struct(node, source_bytes, file_path, parent_module=parent_module)
            if chunk:
                chunks.append(chunk)
        elif node.type == "enum_item":
            chunk = self._extract_enum(node, source_bytes, file_path, parent_module=parent_module)
            if chunk:
                chunks.append(chunk)
        elif node.type == "trait_item":
            chunk = self._extract_trait(node, source_bytes, file_path, parent_module=parent_module)
            if chunk:
                chunks.append(chunk)
        elif node.type == "impl_item":
            chunks.extend(self._extract_impl(node, source_bytes, file_path, parent_module=parent_module))
        elif node.type in ("const_item", "static_item"):
            chunk = self._extract_constant(node, source_bytes, file_path, parent_module=parent_module)
            if chunk:
                chunks.append(chunk)
        elif node.type == "type_item":
            chunk = self._extract_type_alias(node, source_bytes, file_path, parent_module=parent_module)
            if chunk:
                chunks.append(chunk)
        elif node.type == "mod_item":
            chunks.extend(self._extract_mod(node, source_bytes, file_path))
        elif node.type == "macro_definition":
            chunk = self._extract_macro(node, source_bytes, file_path, parent_module=parent_module)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _extract_macro(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        parent_module: Optional[str] = None,
    ) -> Optional[CodeChunk]:
        name = self._get_name(node, source_bytes)
        if not name:
            return None

        line_start, line_end = self.get_node_line_range(node)
        content = self.get_node_text(node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="macro",
            symbol_name=name,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            parent_module=parent_module,
            visibility="public",
        )
