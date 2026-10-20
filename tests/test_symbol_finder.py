"""Tests for the multi-language symbol finder module."""

import pytest
from mullet.core.knowledge.symbol_finder import (
    SymbolMatch,
    list_symbols,
    find_symbol,
    find_symbols_by_type,
    find_all_symbols,
    detect_language,
    SUPPORTED_LANGUAGES,
)


# ── Sample source files ─────────────────────────────────────────────────────

PYTHON_SAMPLE = '''\
class MyClass:
    def __init__(self):
        pass

    def my_method(self, arg1, arg2):
        pass

    async def async_method(self, data):
        pass

async def async_func(param):
    pass

def regular_func(data: str) -> bool:
    return True

class AnotherClass(MyClass):
    pass
'''

TS_SAMPLE = '''\
export class UserService extends BaseService implements Cacheable {
    private cache: Map<string, User>;

    constructor(private db: Database) {
        super();
    }

    async getUser(id: string): Promise<User> {
        return this.db.find(id);
    }

    static fromConfig(config: Config): UserService {
        return new UserService(config.db);
    }
}

export interface UserDTO {
    id: string;
    name: string;
}

export type UserId = string | number;

export enum Role {
    Admin,
    User,
    Guest,
}

export async function fetchUsers(limit: number): Promise<User[]> {
    return [];
}

const processUser = (user: User) => {
    return user;
};

export const asyncHelper = async (data: string) => {
    return data;
};
'''

GO_SAMPLE = '''\
package main

type Server struct {
    addr string
    port int
}

type Handler interface {
    Handle(req Request) Response
}

type StringSet = map[string]bool

func NewServer(addr string, port int) *Server {
    return &Server{addr: addr, port: port}
}

func (s *Server) Start() error {
    return nil
}

func (s *Server) handleRequest(w Writer, r *Request) {
    w.Write([]byte("ok"))
}

func main() {
    s := NewServer("localhost", 8080)
    s.Start()
}
'''

PHP_SAMPLE = '''\
<?php

abstract class BaseController {
    protected $db;

    public function __construct(Database $db) {
        $this->db = $db;
    }

    abstract public function index();
}

class UserController extends BaseController implements Cacheable {
    public function index() {
        return $this->db->query("SELECT * FROM users");
    }

    private function validateUser($data) {
        return true;
    }

    public static function create($data) {
        return new self($data);
    }
}

interface Cacheable {
    public function getCacheKey();
}

trait Loggable {
    public function log($message) {
        echo $message;
    }
}

function helper_function($param) {
    return $param;
}
'''


@pytest.fixture
def sample_tree(tmp_path):
    """Create a multi-language project tree."""
    (tmp_path / "main.py").write_text(PYTHON_SAMPLE)
    (tmp_path / "service.ts").write_text(TS_SAMPLE)
    (tmp_path / "server.go").write_text(GO_SAMPLE)
    (tmp_path / "controller.php").write_text(PHP_SAMPLE)
    (tmp_path / "README.md").write_text("Not code.")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.py").write_text("def nested_func(): pass\n")
    (sub / "util.ts").write_text("export function utilFunc(): void {}\n")
    return tmp_path


# ── SymbolMatch dataclass ───────────────────────────────────────────────────

def test_symbol_match_fields():
    s = SymbolMatch(name="foo", type="function", file_path="/a.py",
                    line_number=5, signature="def foo():")
    assert s.name == "foo"
    assert s.parent_class is None


def test_symbol_match_with_parent():
    s = SymbolMatch(name="bar", type="method", file_path="/a.py",
                    line_number=10, signature="def bar(self):",
                    parent_class="Baz")
    assert s.parent_class == "Baz"


# ── detect_language ─────────────────────────────────────────────────────────

def test_detect_language_python():
    assert detect_language("foo.py") == "python"

def test_detect_language_ts():
    assert detect_language("bar.ts") == "typescript"
    assert detect_language("baz.tsx") == "typescript"

def test_detect_language_go():
    assert detect_language("main.go") == "go"

def test_detect_language_php():
    assert detect_language("index.php") == "php"
    assert detect_language("mymod.module") == "php"

def test_detect_language_unknown():
    assert detect_language("file.rs") is None


# ── Python ──────────────────────────────────────────────────────────────────

def test_py_classes(sample_tree):
    syms = list_symbols(str(sample_tree / "main.py"), "python")
    classes = [s for s in syms if s.type == "class"]
    assert {c.name for c in classes} == {"MyClass", "AnotherClass"}


def test_py_methods(sample_tree):
    syms = list_symbols(str(sample_tree / "main.py"), "python")
    methods = [s for s in syms if s.type == "method"]
    assert {m.name for m in methods} == {"__init__", "my_method"}
    for m in methods:
        assert m.parent_class == "MyClass"


def test_py_async_method(sample_tree):
    syms = list_symbols(str(sample_tree / "main.py"), "python")
    async_methods = [s for s in syms if s.type == "async_method"]
    assert len(async_methods) == 1
    assert async_methods[0].name == "async_method"
    assert async_methods[0].parent_class == "MyClass"


def test_py_functions(sample_tree):
    syms = list_symbols(str(sample_tree / "main.py"), "python")
    funcs = [s for s in syms if s.type == "function"]
    assert len(funcs) == 1
    assert funcs[0].name == "regular_func"
    assert "-> bool" in funcs[0].signature


def test_py_async_function(sample_tree):
    syms = list_symbols(str(sample_tree / "main.py"), "python")
    async_funcs = [s for s in syms if s.type == "async_function"]
    assert len(async_funcs) == 1
    assert async_funcs[0].name == "async_func"


def test_py_line_numbers(sample_tree):
    syms = list_symbols(str(sample_tree / "main.py"), "python")
    by_name = {s.name: s.line_number for s in syms}
    assert by_name["MyClass"] == 1
    assert by_name["AnotherClass"] == 17


def test_py_missing_file():
    assert list_symbols("/nonexistent.py") == []


# ── TypeScript ──────────────────────────────────────────────────────────────

def test_ts_class(sample_tree):
    syms = list_symbols(str(sample_tree / "service.ts"), "typescript")
    classes = [s for s in syms if s.type == "class"]
    assert len(classes) == 1
    assert classes[0].name == "UserService"


def test_ts_interface(sample_tree):
    syms = list_symbols(str(sample_tree / "service.ts"), "typescript")
    ifaces = [s for s in syms if s.type == "interface"]
    assert len(ifaces) == 1
    assert ifaces[0].name == "UserDTO"


def test_ts_type_alias(sample_tree):
    syms = list_symbols(str(sample_tree / "service.ts"), "typescript")
    types = [s for s in syms if s.type == "type_alias"]
    assert len(types) == 1
    assert types[0].name == "UserId"


def test_ts_enum(sample_tree):
    syms = list_symbols(str(sample_tree / "service.ts"), "typescript")
    enums = [s for s in syms if s.type == "enum"]
    assert len(enums) == 1
    assert enums[0].name == "Role"


def test_ts_function(sample_tree):
    syms = list_symbols(str(sample_tree / "service.ts"), "typescript")
    funcs = [s for s in syms if s.type in ("function", "async_function")]
    names = {f.name for f in funcs}
    assert "fetchUsers" in names
    assert "processUser" in names
    assert "asyncHelper" in names


def test_ts_methods(sample_tree):
    syms = list_symbols(str(sample_tree / "service.ts"), "typescript")
    methods = [s for s in syms if s.type in ("method", "async_method")]
    names = {m.name for m in methods}
    assert "getUser" in names
    assert "fromConfig" in names
    for m in methods:
        assert m.parent_class == "UserService"


def test_ts_find_across_files(sample_tree):
    matches = find_symbol("utilFunc", str(sample_tree), "typescript")
    assert len(matches) == 1
    assert "util.ts" in matches[0].file_path


# ── Go ──────────────────────────────────────────────────────────────────────

def test_go_struct(sample_tree):
    syms = list_symbols(str(sample_tree / "server.go"), "go")
    structs = [s for s in syms if s.type == "struct"]
    assert len(structs) == 1
    assert structs[0].name == "Server"


def test_go_interface(sample_tree):
    syms = list_symbols(str(sample_tree / "server.go"), "go")
    ifaces = [s for s in syms if s.type == "interface"]
    assert len(ifaces) == 1
    assert ifaces[0].name == "Handler"


def test_go_type_alias(sample_tree):
    syms = list_symbols(str(sample_tree / "server.go"), "go")
    aliases = [s for s in syms if s.type == "type_alias"]
    assert len(aliases) == 1
    assert aliases[0].name == "StringSet"


def test_go_functions(sample_tree):
    syms = list_symbols(str(sample_tree / "server.go"), "go")
    funcs = [s for s in syms if s.type == "function"]
    names = {f.name for f in funcs}
    assert "NewServer" in names
    assert "main" in names


def test_go_methods(sample_tree):
    syms = list_symbols(str(sample_tree / "server.go"), "go")
    methods = [s for s in syms if s.type == "method"]
    names = {m.name for m in methods}
    assert "Start" in names
    assert "handleRequest" in names
    for m in methods:
        assert m.parent_class == "Server"


# ── PHP ─────────────────────────────────────────────────────────────────────

def test_php_classes(sample_tree):
    syms = list_symbols(str(sample_tree / "controller.php"), "php")
    classes = [s for s in syms if s.type == "class"]
    names = {c.name for c in classes}
    assert "BaseController" in names
    assert "UserController" in names


def test_php_interface(sample_tree):
    syms = list_symbols(str(sample_tree / "controller.php"), "php")
    ifaces = [s for s in syms if s.type == "interface"]
    assert len(ifaces) == 1
    assert ifaces[0].name == "Cacheable"


def test_php_trait(sample_tree):
    syms = list_symbols(str(sample_tree / "controller.php"), "php")
    traits = [s for s in syms if s.type == "trait"]
    assert len(traits) == 1
    assert traits[0].name == "Loggable"


def test_php_methods(sample_tree):
    syms = list_symbols(str(sample_tree / "controller.php"), "php")
    methods = [s for s in syms if s.type == "method"]
    names = {m.name for m in methods}
    assert "__construct" in names
    assert "index" in names
    assert "validateUser" in names
    assert "create" in names
    assert "getCacheKey" in names
    assert "log" in names


def test_php_function(sample_tree):
    syms = list_symbols(str(sample_tree / "controller.php"), "php")
    funcs = [s for s in syms if s.type == "function"]
    assert len(funcs) == 1
    assert funcs[0].name == "helper_function"


def test_php_method_parent_class(sample_tree):
    syms = list_symbols(str(sample_tree / "controller.php"), "php")
    construct = next(s for s in syms if s.name == "__construct")
    assert construct.parent_class == "BaseController"
    validate = next(s for s in syms if s.name == "validateUser")
    assert validate.parent_class == "UserController"


# ── Cross-language features ─────────────────────────────────────────────────

def test_find_symbol_fuzzy(sample_tree):
    matches = find_symbol("func", str(sample_tree), "python", exact=False)
    names = {m.name for m in matches}
    assert "regular_func" in names
    assert "async_func" in names
    assert "nested_func" in names


def test_find_symbol_max_results(sample_tree):
    matches = find_symbol("func", str(sample_tree), "python", exact=False, max_results=2)
    assert len(matches) == 2


def test_find_symbols_by_type_go(sample_tree):
    structs = find_symbols_by_type(str(sample_tree), "struct", "go")
    assert len(structs) == 1
    assert structs[0].name == "Server"


def test_find_all_symbols_single_lang(sample_tree):
    syms = find_all_symbols(str(sample_tree), language="python")
    names = {s.name for s in syms}
    assert "MyClass" in names
    assert "nested_func" in names


def test_find_all_symbols_all_langs(sample_tree):
    syms = find_all_symbols(str(sample_tree))
    names = {s.name for s in syms}
    # Should have symbols from every language
    assert "MyClass" in names        # Python
    assert "UserService" in names    # TypeScript
    assert "Server" in names         # Go
    assert "BaseController" in names # PHP


def test_unsupported_language():
    with pytest.raises(ValueError, match="Unsupported language"):
        list_symbols("/x.rs", language="rust")
    with pytest.raises(ValueError, match="Unsupported language"):
        find_symbol("foo", "/tmp", language="rust")


def test_supported_languages_set():
    assert "python" in SUPPORTED_LANGUAGES
    assert "typescript" in SUPPORTED_LANGUAGES
    assert "go" in SUPPORTED_LANGUAGES
    assert "php" in SUPPORTED_LANGUAGES


def test_skip_hidden_dirs(tmp_path):
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.py").write_text("def secret(): pass\n")
    (tmp_path / "visible.py").write_text("def visible(): pass\n")

    matches = find_symbol("secret", str(tmp_path))
    assert matches == []
    matches = find_symbol("visible", str(tmp_path))
    assert len(matches) == 1


def test_skip_node_modules(tmp_path):
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "lib.ts").write_text("export function lib(): void {}\n")
    (tmp_path / "app.ts").write_text("export function app(): void {}\n")

    matches = find_symbol("lib", str(tmp_path), "typescript")
    assert matches == []
    matches = find_symbol("app", str(tmp_path), "typescript")
    assert len(matches) == 1


def test_skip_vendor_dir(tmp_path):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "dep.php").write_text("<?php\nfunction vendored() {}\n")
    (tmp_path / "app.php").write_text("<?php\nfunction myApp() {}\n")

    matches = find_symbol("vendored", str(tmp_path), "php")
    assert matches == []
    matches = find_symbol("myApp", str(tmp_path), "php")
    assert len(matches) == 1
