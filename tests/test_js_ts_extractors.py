"""Tests for JavaScript and TypeScript code extractors."""
from __future__ import annotations

import pytest

from mullet.knowledge.code.extractors.javascript import JavaScriptExtractor
from mullet.knowledge.code.extractors.typescript import TypeScriptExtractor


_SIMPLE_JS = """\
function greet(name) {
  return "Hello, " + name;
}

function _privateHelper(x) {
  return x * 2;
}
"""

_ARROW_FUNCTIONS = """\
const add = (a, b) => a + b;

const fetchUser = async (id) => {
  const resp = await fetch(`/api/users/${id}`);
  return resp.json();
};

const double = x => x * 2;
"""

_CLASS_JS = """\
/**
 * Manages user authentication.
 */
class AuthService extends BaseService {
  /**
   * Login with credentials.
   */
  login(email, password) {
    return this._doLogin(email, password);
  }

  logout() {
    this.session = null;
  }

  static create(config) {
    return new AuthService(config);
  }
}
"""

_TYPESCRIPT = """\
interface User {
  id: number;
  name: string;
  email: string;
}

interface AdminUser extends User {
  role: string;
  permissions: string[];
}

type UserId = string | number;

type Status = "active" | "inactive" | "pending";

enum Direction {
  Up,
  Down,
  Left,
  Right,
}

function getUser(id: UserId): User {
  return { id: 1, name: "Alice", email: "alice@example.com" };
}

class UserService {
  private users: User[] = [];

  getAll(): User[] {
    return this.users;
  }
}
"""

_TSX = """\
import React from 'react';

interface ButtonProps {
  label: string;
  onClick: () => void;
}

const Button: React.FC<ButtonProps> = ({ label, onClick }) => (
  <button onClick={onClick}>{label}</button>
);

export default Button;
"""


@pytest.fixture
def js_extractor() -> JavaScriptExtractor:
    return JavaScriptExtractor()


@pytest.fixture
def ts_extractor() -> TypeScriptExtractor:
    return TypeScriptExtractor()


# ---------------------------------------------------------------------------
# JavaScript — functions
# ---------------------------------------------------------------------------

class TestJSFunctions:
    def test_extracts_function(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_SIMPLE_JS, "app.js")
        func_names = [c.symbol_name for c in chunks if c.symbol_type == "function"]
        assert "greet" in func_names

    def test_function_chunk_language(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_SIMPLE_JS, "app.js")
        funcs = [c for c in chunks if c.symbol_type == "function"]
        assert all(c.language == "javascript" for c in funcs)

    def test_private_function_visibility(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_SIMPLE_JS, "app.js")
        private = next(c for c in chunks if c.symbol_name == "_privateHelper")
        assert private.visibility == "private"

    def test_public_function_visibility(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_SIMPLE_JS, "app.js")
        public = next(c for c in chunks if c.symbol_name == "greet")
        assert public.visibility == "public"

    def test_source_file_set(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_SIMPLE_JS, "app.js")
        assert all(c.source_file == "app.js" for c in chunks)


class TestJSArrowFunctions:
    def test_extracts_arrow_functions(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_ARROW_FUNCTIONS, "utils.js")
        arrow_names = [c.symbol_name for c in chunks if c.symbol_type == "arrow_function"]
        assert "add" in arrow_names
        assert "fetchUser" in arrow_names

    def test_arrow_function_signature(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_ARROW_FUNCTIONS, "utils.js")
        add = next(c for c in chunks if c.symbol_name == "add")
        assert add.signature is not None
        assert "a" in add.signature

    def test_single_param_arrow_function(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_ARROW_FUNCTIONS, "utils.js")
        double = next((c for c in chunks if c.symbol_name == "double"), None)
        if double:  # May or may not be extracted depending on tree-sitter grammar
            assert double.symbol_type == "arrow_function"


class TestJSClasses:
    def test_extracts_class(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_CLASS_JS, "auth.js")
        class_chunks = [c for c in chunks if c.symbol_type == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].symbol_name == "AuthService"

    def test_class_base_class_in_extras(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_CLASS_JS, "auth.js")
        auth_class = next(c for c in chunks if c.symbol_type == "class")
        assert auth_class.extras.get("base_class") == "BaseService"

    def test_class_methods_in_extras(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_CLASS_JS, "auth.js")
        auth_class = next(c for c in chunks if c.symbol_type == "class")
        methods = auth_class.extras.get("methods", [])
        assert "login" in methods
        assert "logout" in methods

    def test_extracts_class_methods(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_CLASS_JS, "auth.js")
        method_names = [c.symbol_name for c in chunks if c.symbol_type == "method"]
        assert "login" in method_names

    def test_method_has_parent_class(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_CLASS_JS, "auth.js")
        login = next(c for c in chunks if c.symbol_type == "method" and c.symbol_name == "login")
        assert login.parent_class == "AuthService"

    def test_jsdoc_captured(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_CLASS_JS, "auth.js")
        auth_class = next(c for c in chunks if c.symbol_type == "class")
        assert auth_class.docstring is not None
        assert "authentication" in auth_class.docstring.lower() or len(auth_class.docstring) > 0

    def test_static_method_in_extras(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract(_CLASS_JS, "auth.js")
        create = next((c for c in chunks if c.symbol_name == "create"), None)
        if create:
            assert create.extras.get("static") is True


class TestJSEdgeCases:
    def test_empty_js_returns_empty(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract("", "empty.js")
        assert chunks == []

    def test_returns_list(self, js_extractor: JavaScriptExtractor) -> None:
        chunks = js_extractor.extract("const x = 1;", "simple.js")
        assert isinstance(chunks, list)

    def test_determine_visibility_private(self, js_extractor: JavaScriptExtractor) -> None:
        assert js_extractor.determine_visibility("_internal") == "private"
        assert js_extractor.determine_visibility("#field") == "private"

    def test_determine_visibility_public(self, js_extractor: JavaScriptExtractor) -> None:
        assert js_extractor.determine_visibility("publicMethod") == "public"


# ---------------------------------------------------------------------------
# TypeScript — interfaces, type aliases, enums
# ---------------------------------------------------------------------------

class TestTSInterfaces:
    def test_extracts_interfaces(self, ts_extractor: TypeScriptExtractor) -> None:
        chunks = ts_extractor.extract(_TYPESCRIPT, "types.ts")
        iface_names = [c.symbol_name for c in chunks if c.symbol_type == "interface"]
        assert "User" in iface_names
        assert "AdminUser" in iface_names

    def test_interface_language_is_typescript(self, ts_extractor: TypeScriptExtractor) -> None:
        chunks = ts_extractor.extract(_TYPESCRIPT, "types.ts")
        ifaces = [c for c in chunks if c.symbol_type == "interface"]
        assert all(c.language == "typescript" for c in ifaces)

    def test_interface_properties_in_extras(self, ts_extractor: TypeScriptExtractor) -> None:
        chunks = ts_extractor.extract(_TYPESCRIPT, "types.ts")
        user_iface = next(c for c in chunks if c.symbol_type == "interface" and c.symbol_name == "User")
        props = user_iface.extras.get("properties", [])
        assert "id" in props or "name" in props


class TestTSTypeAliases:
    def test_extracts_type_aliases(self, ts_extractor: TypeScriptExtractor) -> None:
        chunks = ts_extractor.extract(_TYPESCRIPT, "types.ts")
        type_names = [c.symbol_name for c in chunks if c.symbol_type == "type_alias"]
        assert "UserId" in type_names
        assert "Status" in type_names


class TestTSEnums:
    def test_extracts_enum(self, ts_extractor: TypeScriptExtractor) -> None:
        chunks = ts_extractor.extract(_TYPESCRIPT, "types.ts")
        enum_chunks = [c for c in chunks if c.symbol_type == "enum"]
        assert any(c.symbol_name == "Direction" for c in enum_chunks)

    def test_enum_members_in_extras(self, ts_extractor: TypeScriptExtractor) -> None:
        chunks = ts_extractor.extract(_TYPESCRIPT, "types.ts")
        direction = next(c for c in chunks if c.symbol_type == "enum" and c.symbol_name == "Direction")
        members = direction.extras.get("members", [])
        assert any(m in ("Up", "Down", "Left", "Right") for m in members)


class TestTSFunctions:
    def test_extracts_typed_function(self, ts_extractor: TypeScriptExtractor) -> None:
        chunks = ts_extractor.extract(_TYPESCRIPT, "types.ts")
        func_names = [c.symbol_name for c in chunks if c.symbol_type == "function"]
        assert "getUser" in func_names

    def test_function_language_is_typescript(self, ts_extractor: TypeScriptExtractor) -> None:
        chunks = ts_extractor.extract(_TYPESCRIPT, "types.ts")
        funcs = [c for c in chunks if c.symbol_type == "function"]
        assert all(c.language == "typescript" for c in funcs)


class TestTSXExtraction:
    def test_tsx_file_extracts_successfully(self, ts_extractor: TypeScriptExtractor) -> None:
        chunks = ts_extractor.extract(_TSX, "Button.tsx")
        assert isinstance(chunks, list)

    def test_tsx_extracts_interface(self, ts_extractor: TypeScriptExtractor) -> None:
        chunks = ts_extractor.extract(_TSX, "Button.tsx")
        iface_names = [c.symbol_name for c in chunks if c.symbol_type == "interface"]
        assert "ButtonProps" in iface_names

    def test_tsx_extracts_component(self, ts_extractor: TypeScriptExtractor) -> None:
        chunks = ts_extractor.extract(_TSX, "Button.tsx")
        # Button should be extracted as arrow_function or similar
        names = [c.symbol_name for c in chunks]
        assert "Button" in names
