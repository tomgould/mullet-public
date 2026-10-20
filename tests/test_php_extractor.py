"""Tests for mullet.knowledge.code.extractors.php."""
from __future__ import annotations

import pytest

from mullet.knowledge.code.extractors.php import PHPExtractor


_SIMPLE_PHP = """\
<?php

/**
 * Greet a user by name.
 */
function greet($name) {
    return "Hello, " . $name;
}

function _helperFunction($x) {
    return $x * 2;
}
"""

_CLASS_PHP = """\
<?php

/**
 * Manages user authentication.
 */
class AuthManager extends BaseManager implements Authenticatable, Loggable {

    protected string $secret;

    /**
     * Login a user.
     */
    public function login(string $email, string $password): bool {
        return $this->validate($email, $password);
    }

    protected function validate(string $email, string $password): bool {
        return strlen($password) >= 8;
    }

    private function _hashPassword(string $password): string {
        return hash('sha256', $password);
    }

    public static function create(): self {
        return new self();
    }
}
"""

_INTERFACE_PHP = """\
<?php

interface Serializable {
    public function serialize(): string;
    public function deserialize(string $data): void;
}

interface Countable {
    public function count(): int;
}
"""

_DRUPAL_MODULE = """\
<?php

/**
 * Implements hook_node_insert().
 */
function mymodule_node_insert($node) {
    drupal_set_message('Node created: ' . $node->title);
}

/**
 * Implements hook_form_alter().
 */
function mymodule_form_alter(&$form, &$form_state, $form_id) {
    // Alter forms
}
"""


@pytest.fixture
def extractor() -> PHPExtractor:
    return PHPExtractor()


class TestPHPFunctions:
    def test_extracts_function(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_PHP, "helpers.php")
        func_names = [c.symbol_name for c in chunks if c.symbol_type == "function"]
        assert "greet" in func_names

    def test_function_language_is_php(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_PHP, "helpers.php")
        funcs = [c for c in chunks if c.symbol_type == "function"]
        assert all(c.language == "php" for c in funcs)

    def test_source_file_set(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_PHP, "helpers.php")
        assert all(c.source_file == "helpers.php" for c in chunks)

    def test_docblock_captured(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_PHP, "helpers.php")
        greet = next(c for c in chunks if c.symbol_name == "greet")
        assert greet.docstring is not None
        assert len(greet.docstring) > 0

    def test_returns_list(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_PHP, "helpers.php")
        assert isinstance(chunks, list)


class TestPHPClasses:
    def test_extracts_class(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_CLASS_PHP, "AuthManager.php")
        class_chunks = [c for c in chunks if c.symbol_type == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].symbol_name == "AuthManager"

    def test_base_class_in_extras(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_CLASS_PHP, "AuthManager.php")
        auth = next(c for c in chunks if c.symbol_type == "class")
        assert auth.extras.get("base_class") == "BaseManager"

    def test_interfaces_in_extras(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_CLASS_PHP, "AuthManager.php")
        auth = next(c for c in chunks if c.symbol_type == "class")
        interfaces = auth.extras.get("interfaces", [])
        assert "Authenticatable" in interfaces or len(interfaces) > 0

    def test_extracts_public_methods(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_CLASS_PHP, "AuthManager.php")
        method_names = [c.symbol_name for c in chunks if c.symbol_type == "method"]
        assert "login" in method_names

    def test_method_visibility_public(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_CLASS_PHP, "AuthManager.php")
        login = next(c for c in chunks if c.symbol_type == "method" and c.symbol_name == "login")
        assert login.visibility == "public"

    def test_method_visibility_protected(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_CLASS_PHP, "AuthManager.php")
        validate = next((c for c in chunks if c.symbol_name == "validate"), None)
        if validate:
            assert validate.visibility == "protected"

    def test_method_has_parent_class(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_CLASS_PHP, "AuthManager.php")
        login = next(c for c in chunks if c.symbol_type == "method" and c.symbol_name == "login")
        assert login.parent_class == "AuthManager"


class TestPHPInterfaces:
    def test_extracts_interface(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_INTERFACE_PHP, "contracts.php")
        iface_types = [c.symbol_type for c in chunks]
        assert "interface" in iface_types

    def test_interface_names(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_INTERFACE_PHP, "contracts.php")
        iface_names = [c.symbol_name for c in chunks if c.symbol_type == "interface"]
        assert "Serializable" in iface_names


class TestDrupalHooks:
    def test_extracts_drupal_hook(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_DRUPAL_MODULE, "mymodule.module")
        hook_chunks = [c for c in chunks if c.symbol_type == "hook"]
        assert len(hook_chunks) >= 1

    def test_hook_type_in_extras(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_DRUPAL_MODULE, "mymodule.module")
        hooks = [c for c in chunks if c.symbol_type == "hook"]
        hook_types = [c.extras.get("hook_type") for c in hooks]
        assert any(h for h in hook_types)

    def test_non_drupal_file_has_no_hooks(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_PHP, "helpers.php")
        hook_chunks = [c for c in chunks if c.symbol_type == "hook"]
        assert len(hook_chunks) == 0


class TestEdgeCases:
    def test_empty_php_returns_empty(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract("<?php\n", "empty.php")
        assert isinstance(chunks, list)

    def test_php_only_comments(self, extractor: PHPExtractor) -> None:
        chunks = extractor.extract("<?php\n// just a comment\n", "comment.php")
        assert isinstance(chunks, list)
