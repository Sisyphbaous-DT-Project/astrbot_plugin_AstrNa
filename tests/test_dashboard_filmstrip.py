"""胶卷滚轮手势状态机测试：Python 启动 Node 直接驱动 wheel-gesture.js 纯模块。"""

import base64
import shutil
import subprocess
from pathlib import Path

import pytest

GESTURE_MODULE = (
    Path(__file__).resolve().parent.parent
    / "pages"
    / "dashboard"
    / "js"
    / "wheel-gesture.js"
)


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 Node.js 执行滚轮手势状态机")
def test_wheel_gesture_state_machine():
    source = base64.b64encode(GESTURE_MODULE.read_bytes()).decode("ascii")
    script = rf"""
      import assert from "node:assert/strict";
      const moduleUrl = "data:text/javascript;base64,{source}";
      const {{ createWheelGesture, normalizeWheelDelta }} = await import(moduleUrl);

      // 完整管线：浏览承诺一帧 → 手势结束 → 吸附完成 → 静止后武装
      function arm(g, now = 0) {{
        const e = g.wheel({{ delta: 100, now, overCentered: false }});
        assert.equal(e.openDetail, false);
        assert.equal(e.commits, 1);
        assert.equal(g.endGesture().snap, "toTarget");
        const done = g.snapDone({{ centered: true, switched: true }});
        assert.equal(done.armAfter, 300);
        assert.deepEqual(g.quietElapsed(done.token), {{ armed: true }});
        assert.equal(g.isArmed(), true);
      }}

      // 1. 初始单个 deltaY=200 只能浏览，绝不能打开详情
      {{
        const g = createWheelGesture();
        const e = g.wheel({{ delta: 200, now: 0, overCentered: true }});
        assert.equal(e.openDetail, false);
        assert.equal(e.commits, 2);
        assert.equal(e.scrollBy, 200);
      }}

      // 2. 约 70px 的轻滚稳定承诺一帧（含拆成两个事件的累计）
      {{
        const g1 = createWheelGesture();
        assert.equal(g1.wheel({{ delta: 70, now: 0 }}).commits, 1);
        const g2 = createWheelGesture();
        assert.equal(g2.wheel({{ delta: 40, now: 0 }}).commits, 0);
        assert.equal(g2.wheel({{ delta: 40, now: 60 }}).commits, 1);
      }}

      // 3. 同一连续手势可以推进多帧，并且始终不进入详情
      {{
        const g = createWheelGesture();
        for (let i = 0; i < 4; i += 1) {{
          const e = g.wheel({{ delta: 200, now: i * 80, overCentered: true }});
          assert.equal(e.openDetail, false);
          assert.ok(e.commits >= 2);
        }}
        assert.equal(g.phase, "gesture");
      }}

      // 4. 未达到门槛时吸回原帧
      {{
        const g = createWheelGesture();
        assert.equal(g.wheel({{ delta: 30, now: 0 }}).commits, 0);
        assert.equal(g.endGesture().snap, "back");
      }}

      // 5. 吸附动画未完成时不能打开详情
      {{
        const g = createWheelGesture();
        arm(g);
        g.setSnapActive(true); // 新一轮吸附进行中
        const e = g.wheel({{ delta: 200, now: 1000, overCentered: true }});
        assert.equal(e.openDetail, false);
      }}

      // 6. 完成真实翻页、居中、静止后，新的向前手势才允许打开详情
      {{
        const g = createWheelGesture();
        arm(g);
        const e = g.wheel({{ delta: 50, now: 1000, overCentered: true, detailOpen: false }});
        assert.equal(e.openDetail, true);
        // 触发后武装失效，后续事件只能浏览
        const again = g.wheel({{ delta: 50, now: 1050, overCentered: true }});
        assert.equal(again.openDetail, false);
      }}

      // 6b. 吸附未居中 / 未真实换帧时不武装
      {{
        const g = createWheelGesture();
        g.wheel({{ delta: 100, now: 0 }});
        g.endGesture();
        const bad = g.snapDone({{ centered: false, switched: true }});
        assert.equal(bad.armAfter, null);
        assert.deepEqual(g.quietElapsed(bad.token), {{ armed: false }});
        const g2 = createWheelGesture();
        g2.wheel({{ delta: 10, now: 0 }});
        g2.endGesture();
        const noSwitch = g2.snapDone({{ centered: true, switched: false }});
        assert.deepEqual(g2.quietElapsed(noSwitch.token), {{ armed: false }});
      }}

      // 7. 向后滚动不进入详情
      {{
        const g = createWheelGesture();
        arm(g);
        const e = g.wheel({{ delta: -120, now: 1000, overCentered: true }});
        assert.equal(e.openDetail, false);
        assert.equal(g.isArmed(), false); // 转为浏览手势
      }}

      // 7b. 指针不在居中帧 / 详情已打开时不进入详情
      {{
        const g = createWheelGesture();
        arm(g);
        assert.equal(g.wheel({{ delta: 100, now: 1000, overCentered: false }}).openDetail, false);
        const g2 = createWheelGesture();
        arm(g2);
        assert.equal(
          g2.wheel({{ delta: 100, now: 1000, overCentered: true, detailOpen: true }}).openDetail,
          false,
        );
      }}

      // 8. deltaMode 三种模式统一：像素/行/页归一化后行为一致
      {{
        assert.equal(normalizeWheelDelta({{ deltaY: 80, deltaMode: 0 }}), 80);
        assert.equal(normalizeWheelDelta({{ deltaY: 5, deltaMode: 1 }}), 80);
        assert.equal(normalizeWheelDelta({{ deltaY: 1, deltaMode: 2, pageSize: 600 }}), 600);
        assert.equal(normalizeWheelDelta({{ deltaX: -120, deltaY: 30, deltaMode: 0 }}), -120);
        const px = createWheelGesture();
        const lines = createWheelGesture();
        const pxDelta = normalizeWheelDelta({{ deltaY: 80, deltaMode: 0 }});
        const lineDelta = normalizeWheelDelta({{ deltaY: 5, deltaMode: 1 }});
        assert.equal(px.wheel({{ delta: pxDelta, now: 0 }}).commits, 1);
        assert.equal(lines.wheel({{ delta: lineDelta, now: 0 }}).commits, 1);
      }}

      // 9. 从详情返回后旧武装状态失效，第一批滚轮只能浏览
      {{
        const g = createWheelGesture();
        arm(g);
        g.detailClosed();
        const e = g.wheel({{ delta: 100, now: 2000, overCentered: true }});
        assert.equal(e.openDetail, false);
        assert.equal(g.isArmed(), false);
      }}

      // 10. disarm（按钮/方向键/拖动导航）清除手势与武装，无过期状态残留
      {{
        const g = createWheelGesture();
        arm(g);
        g.disarm();
        assert.equal(g.phase, "unarmed");
        assert.equal(g.isArmed(), false);
        const e = g.wheel({{ delta: 100, now: 5000, overCentered: true }});
        assert.equal(e.openDetail, false);
        assert.equal(e.commits, 1);
      }}

      // 11. 手势中途方向反转不沿用残留力度
      {{
        const g = createWheelGesture();
        assert.equal(g.wheel({{ delta: 100, now: 0 }}).commits, 1);
        const e = g.wheel({{ delta: -100, now: 60 }});
        assert.equal(e.commits, -1); // 而不是正负相消后的 0
      }}

      // 12. 单个异常大事件最多承诺 maxFramesPerEvent 帧
      {{
        const g = createWheelGesture();
        const e = g.wheel({{ delta: 1000, now: 0 }});
        assert.equal(e.commits, 3);
        assert.equal(e.scrollBy, 1000); // 超出部分仍作即时位移反馈
      }}

      // 13. 静止期内的滚轮事件使旧武装令牌失效
      {{
        const g = createWheelGesture();
        g.wheel({{ delta: 100, now: 0 }});
        g.endGesture();
        const done = g.snapDone({{ centered: true, switched: true }});
        g.wheel({{ delta: 10, now: 100 }}); // 静止期内又来事件
        assert.deepEqual(g.quietElapsed(done.token), {{ armed: false }});
      }}
    """
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_step_by_uses_latest_target_frame():
    """快速重复按钮/方向键：stepBy 必须以最新目标帧为基准，而非过期帧。"""
    text = GESTURE_MODULE.with_name("filmstrip.js").read_text(encoding="utf-8")
    assert "indexOfKey(targetKey || settledKey || currentKey)" in text
