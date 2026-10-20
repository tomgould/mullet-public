"""Registers and executes pre/post hooks for agent tool execution."""
class HookRegistry:
    """Registry for pre- and post-execution tool hooks."""
    def __init__(self):
        self._pre_hooks = {}
        self._post_hooks = {}

    def register_pre_hook(self, tool_name, callback):
        """
        Registers a pre-execution hook for a given tool.
        The callback will be called before the tool is executed.
        """
        if tool_name not in self._pre_hooks:
            self._pre_hooks[tool_name] = []
        self._pre_hooks[tool_name].append(callback)

    def register_post_hook(self, tool_name, callback):
        """
        Registers a post-execution hook for a given tool.
        The callback will be called after the tool has been executed.
        """
        if tool_name not in self._post_hooks:
            self._post_hooks[tool_name] = []
        self._post_hooks[tool_name].append(callback)

    def execute_pre_hooks(self, tool_name, args):
        """
        Executes all registered pre-hooks for a given tool.
        """
        if tool_name in self._pre_hooks:
            for callback in self._pre_hooks[tool_name]:
                callback(tool_name, args)

    def execute_post_hooks(self, tool_name, args, result):
        """
        Executes all registered post-hooks for a given tool.
        """
        if tool_name in self._post_hooks:
            for callback in self._post_hooks[tool_name]:
                callback(tool_name, args, result)
