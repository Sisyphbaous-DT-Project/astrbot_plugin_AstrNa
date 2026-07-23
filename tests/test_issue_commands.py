from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = REPO_ROOT / "main.py"
ISSUE_OPERATIONS = ("latest", "draft", "ignore", "analyze", "edit", "submit", "cancel")


def _astrna_class_body() -> list:
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AstrNa":
            return node.body
    raise AssertionError("main.py 中缺少 AstrNa 类")


def _astrna_methods() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        item.name: item
        for item in _astrna_class_body()
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _decorator_parts(decorator: ast.expr) -> tuple[str, str, str] | None:
    """返回 (接收者名, 属性名, 首个字符串参数)；不匹配时返回 None。"""
    if not isinstance(decorator, ast.Call) or not decorator.args:
        return None
    func = decorator.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return None
    first_arg = decorator.args[0]
    if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
        return None
    return (func.value.id, func.attr, first_arg.value)


def _command_decorators(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list:
    parts = []
    for decorator in func.decorator_list:
        parsed = _decorator_parts(decorator)
        if parsed is not None and parsed[1] in ("command", "command_group", "group"):
            parts.append(parsed)
    return parts


def test_method_names_are_unique():
    names = [
        item.name
        for item in _astrna_class_body()
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert len(names) == len(set(names))


def test_command_group_structure():
    methods = _astrna_methods()
    root_groups = {}
    subgroup = None
    for name, func in methods.items():
        for receiver, attr, literal in _command_decorators(func):
            if (receiver, attr) == ("filter", "command_group"):
                root_groups[name] = literal
            if (receiver, attr) == ("astrna_command_group", "group"):
                subgroup = (name, literal)
    assert root_groups == {"astrna_command_group": "astrna"}
    assert subgroup == ("astrna_issue_command_group", "issue")
    assert isinstance(methods["astrna_command_group"], ast.FunctionDef)
    assert isinstance(methods["astrna_issue_command_group"], ast.FunctionDef)


def test_grouped_issue_subcommands():
    methods = _astrna_methods()
    grouped = {}
    for name, func in methods.items():
        for receiver, attr, literal in _command_decorators(func):
            if (receiver, attr) == ("astrna_issue_command_group", "command"):
                assert isinstance(func, ast.AsyncFunctionDef), name
                grouped[name] = literal
    assert grouped == {f"issue_{op}_group_command": op for op in ISSUE_OPERATIONS}


def test_underscore_compat_commands():
    methods = _astrna_methods()
    compat = {}
    for name, func in methods.items():
        for receiver, attr, literal in _command_decorators(func):
            if (receiver, attr) == ("filter", "command"):
                assert isinstance(func, ast.AsyncFunctionDef), name
                compat[name] = literal
    assert compat == {f"issue_{op}": f"astrna_issue_{op}" for op in ISSUE_OPERATIONS}


def test_no_spaced_plain_commands_and_no_stacked_command_decorators():
    for name, func in _astrna_methods().items():
        decorators = _command_decorators(func)
        assert len(decorators) <= 1, f"{name} 叠加了多个命令装饰器"
        for receiver, attr, literal in decorators:
            if (receiver, attr) == ("filter", "command"):
                assert " " not in literal, f"{name} 的普通命令名包含空格: {literal!r}"


def test_edit_entries_keep_greedy_str_note():
    methods = _astrna_methods()
    for name in ("issue_edit_group_command", "issue_edit"):
        func = methods[name]
        note_arg = func.args.args[-1]
        assert note_arg.arg == "note", name
        annotation = note_arg.annotation
        assert isinstance(annotation, ast.Name) and annotation.id == "GreedyStr", name


_REAL_REGISTRATION_CODE = """
import asyncio
import importlib.util
import os
import sys
import types

ASTRBOT_SOURCE = __ASTRBOT_SOURCE__
REPO_ROOT = __REPO_ROOT__

sys.path.insert(0, ASTRBOT_SOURCE)
sys.path.insert(0, REPO_ROOT)

from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter
from astrbot.core.star.star_handler import star_handlers_registry

OPS = ["latest", "draft", "ignore", "analyze", "edit", "submit", "cancel"]

# 以包形式加载 main.py，让装饰器真实注册到 AstrBot registry。
pkg = types.ModuleType("astrna_entry")
pkg.__path__ = [REPO_ROOT]
sys.modules["astrna_entry"] = pkg
spec = importlib.util.spec_from_file_location(
    "astrna_entry.main", os.path.join(REPO_ROOT, "main.py")
)
mod = importlib.util.module_from_spec(spec)
sys.modules["astrna_entry.main"] = mod
spec.loader.exec_module(mod)

handlers = [
    md
    for md in star_handlers_registry.star_handlers_map.values()
    if md.handler_module_path == "astrna_entry.main"
]
full_names = [md.handler_full_name for md in handlers]
assert len(set(full_names)) == len(full_names)

group_filters = {}
executable = {}
hook_handlers = []
for md in handlers:
    cmd_filters = [f for f in md.event_filters if isinstance(f, CommandFilter)]
    grp_filters = [f for f in md.event_filters if isinstance(f, CommandGroupFilter)]
    if grp_filters:
        assert len(md.event_filters) == 1, md.handler_full_name
        group_filters[md.handler_name] = grp_filters[0]
        continue
    if cmd_filters:
        assert len(md.event_filters) == 1, md.handler_full_name
        assert len(cmd_filters) == 1, md.handler_full_name
        executable[md.handler_name] = cmd_filters[0]
        continue
    hook_handlers.append(md.handler_name)

assert set(group_filters) == {"astrna_command_group", "astrna_issue_command_group"}
assert len(executable) == 14, sorted(executable)
assert len(hook_handlers) == 8, sorted(hook_handlers)
root_group = group_filters["astrna_command_group"]
issue_group = group_filters["astrna_issue_command_group"]
assert root_group.group_name == "astrna"
assert issue_group.group_name == "issue"
assert issue_group.parent_group is root_group
assert root_group.get_complete_command_names() == ["astrna"]
assert issue_group.get_complete_command_names() == ["astrna issue"]
assert issue_group in root_group.sub_command_filters
sub_names = sorted(
    f.command_name
    for f in issue_group.sub_command_filters
    if isinstance(f, CommandFilter)
)
assert sub_names == sorted(OPS), sub_names

expected_executable = set()
for op in OPS:
    grouped = executable["issue_" + op + "_group_command"]
    assert grouped.command_name == op
    assert grouped.parent_command_names == ["astrna issue"]
    assert grouped.get_complete_command_names() == ["astrna issue " + op]
    compat = executable["issue_" + op]
    assert compat.command_name == "astrna_issue_" + op
    assert compat.parent_command_names == [""]
    assert compat.get_complete_command_names() == ["astrna_issue_" + op]
    expected_executable.add("issue_" + op + "_group_command")
    expected_executable.add("issue_" + op)
assert set(executable) == expected_executable, sorted(executable)


class FakeEvent:
    def __init__(self, text):
        self.is_at_or_wake_command = True
        self.message_str = text
        self.extras = {}

    def get_message_str(self):
        return self.message_str

    def set_extra(self, key, value):
        self.extras[key] = value


# 14 种输入逐一检查：每种输入恰好匹配一个可执行 handler。
inputs = ["astrna issue " + op for op in OPS]
inputs += ["astrna_issue_" + op for op in OPS]
for text in inputs:
    event = FakeEvent(text)
    matched = []
    for name, cmd_filter in executable.items():
        if cmd_filter.filter(event, None):
            matched.append(name)
    assert len(matched) == 1, (text, sorted(matched))
    if text.startswith("astrna issue "):
        expected = "issue_" + text.split()[-1] + "_group_command"
    else:
        expected = "issue_" + text.rsplit("_", 1)[-1]
    assert matched[0] == expected, (text, matched[0], expected)

# 两种 edit 写法都把多词补充内容解析为同一个 GreedyStr。
event_grouped = FakeEvent("astrna issue edit 补充 多词 内容")
assert executable["issue_edit_group_command"].filter(event_grouped, None)
note_grouped = event_grouped.extras["parsed_params"]["note"]
event_compat = FakeEvent("astrna_issue_edit 补充 多词 内容")
assert executable["issue_edit"].filter(event_compat, None)
note_compat = event_compat.extras["parsed_params"]["note"]
assert note_grouped == "补充 多词 内容", note_grouped
assert note_compat == note_grouped, note_compat


def group_tree(group_filter, text):
    try:
        group_filter.filter(FakeEvent(text), None)
    except ValueError as exc:
        return str(exc)
    raise AssertionError("裸组名未触发帮助树")


assert "issue" in group_tree(root_group, "astrna")
issue_tree = group_tree(issue_group, "astrna issue")
for op in OPS:
    assert op in issue_tree, (op, issue_tree)
assert "note" in issue_tree and "GreedyStr" in issue_tree


# 两类入口都调用正确的 Runtime 方法，且每次只发送一次。
class FakeRuntime:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("issue_"):
            raise AttributeError(name)

        async def run(event, *args):
            self.calls.append((name,) + args)
            return "ok-" + name

        return run


class FakeSendEvent:
    def __init__(self):
        self.sent = []

    async def send(self, chain):
        self.sent.append(chain)


inst = mod.AstrNa.__new__(mod.AstrNa)
inst.runtime = FakeRuntime()

for op in OPS:
    for method_name in ("issue_" + op + "_group_command", "issue_" + op):
        inst.runtime.calls.clear()
        send_event = FakeSendEvent()
        method = getattr(inst, method_name)
        if op == "edit":
            asyncio.run(method(send_event, "多词 补充"))
            assert inst.runtime.calls == [("issue_edit", "多词 补充")], (
                method_name,
                inst.runtime.calls,
            )
        else:
            asyncio.run(method(send_event))
            assert inst.runtime.calls == [("issue_" + op,)], (
                method_name,
                inst.runtime.calls,
            )
        assert len(send_event.sent) == 1, method_name

print("ISSUE_COMMANDS_REAL_OK")
"""


def test_real_astrbot_issue_command_registration():
    astrbot_source = os.environ.get("ASTRBOT_SOURCE_PATH")
    if not astrbot_source:
        pytest.skip("未设置 ASTRBOT_SOURCE_PATH")
    if not (Path(astrbot_source) / "astrbot").is_dir():
        pytest.skip("ASTRBOT_SOURCE_PATH 不存在")
    code = _REAL_REGISTRATION_CODE.replace(
        "__ASTRBOT_SOURCE__",
        repr(str(Path(astrbot_source).resolve())),
    ).replace("__REPO_ROOT__", repr(str(REPO_ROOT)))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=180,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ISSUE_COMMANDS_REAL_OK" in result.stdout
