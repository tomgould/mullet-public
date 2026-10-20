from mullet.core.agent.hook import HookRegistry

def test_hook_registry_init():
    registry = HookRegistry()
    assert registry._pre_hooks == {}
    assert registry._post_hooks == {}

def test_register_pre_hook():
    registry = HookRegistry()

    def pre_hook_callback(tool_name, args):
        pass

    registry.register_pre_hook("my_tool", pre_hook_callback)
    assert "my_tool" in registry._pre_hooks
    assert pre_hook_callback in registry._pre_hooks["my_tool"]

def test_register_post_hook():
    registry = HookRegistry()

    def post_hook_callback(tool_name, args, result):
        pass

    registry.register_post_hook("my_tool", post_hook_callback)
    assert "my_tool" in registry._post_hooks
    assert post_hook_callback in registry._post_hooks["my_tool"]

def test_execute_pre_hooks_single_hook():
    registry = HookRegistry()
    called = []

    def pre_hook_callback(tool_name, args):
        called.append((tool_name, args))

    registry.register_pre_hook("my_tool", pre_hook_callback)
    registry.execute_pre_hooks("my_tool", {"arg1": "value1"})
    assert called == [("my_tool", {"arg1": "value1"})]

def test_execute_post_hooks_single_hook():
    registry = HookRegistry()
    called = []

    def post_hook_callback(tool_name, args, result):
        called.append((tool_name, args, result))

    registry.register_post_hook("my_tool", post_hook_callback)
    registry.execute_post_hooks("my_tool", {"arg1": "value1"}, "success")
    assert called == [("my_tool", {"arg1": "value1"}, "success")]

def test_execute_pre_hooks_multiple_hooks():
    registry = HookRegistry()
    called = []

    def pre_hook_callback1(tool_name, args):
        called.append(f"1:{tool_name}:{args}")

    def pre_hook_callback2(tool_name, args):
        called.append(f"2:{tool_name}:{args}")

    registry.register_pre_hook("my_tool", pre_hook_callback1)
    registry.register_pre_hook("my_tool", pre_hook_callback2)
    registry.execute_pre_hooks("my_tool", {"data": 123})
    assert called == [
        "1:my_tool:{'data': 123}",
        "2:my_tool:{'data': 123}"
    ]

def test_execute_post_hooks_multiple_hooks():
    registry = HookRegistry()
    called = []

    def post_hook_callback1(tool_name, args, result):
        called.append(f"1:{tool_name}:{args}:{result}")

    def post_hook_callback2(tool_name, args, result):
        called.append(f"2:{tool_name}:{args}:{result}")

    registry.register_post_hook("my_tool", post_hook_callback1)
    registry.register_post_hook("my_tool", post_hook_callback2)
    registry.execute_post_hooks("my_tool", {"data": 456}, "failure")
    assert called == [
        "1:my_tool:{'data': 456}:failure",
        "2:my_tool:{'data': 456}:failure"
    ]

def test_execute_hooks_no_hooks_registered():
    registry = HookRegistry()
    # Should not raise an error
    registry.execute_pre_hooks("non_existent_tool", {})
    registry.execute_post_hooks("non_existent_tool", {}, None)

def test_register_multiple_tools():
    registry = HookRegistry()
    called_pre = []
    called_post = []

    def pre_hook_tool1(tool_name, args):
        called_pre.append(f"pre1:{tool_name}")

    def post_hook_tool2(tool_name, args, result):
        called_post.append(f"post2:{tool_name}")

    registry.register_pre_hook("tool1", pre_hook_tool1)
    registry.register_post_hook("tool2", post_hook_tool2)

    registry.execute_pre_hooks("tool1", {})
    registry.execute_post_hooks("tool2", {}, "done")

    assert called_pre == ["pre1:tool1"]
    assert called_post == ["post2:tool2"]

def test_pre_hook_modifies_args():
    registry = HookRegistry()

    def pre_hook_modifier(tool_name, args):
        args["modified"] = True

    registry.register_pre_hook("my_tool", pre_hook_modifier)
    args = {"original": True}
    registry.execute_pre_hooks("my_tool", args)
    assert args == {"original": True, "modified": True}
