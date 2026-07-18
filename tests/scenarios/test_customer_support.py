"""Scenario: web customer-support AI — manual search (real sample manuals),
human escalation with transcript capture, chart cards (artifacts相当)."""
from __future__ import annotations

import pytest

from state_projection_loop import ScriptedLLM, Session

from examples.customer_support.tools import (
    SUPPORT_KERNEL,
    SupportBackend,
    build_support_registry,
    load_manuals,
)


@pytest.fixture()
def backend():
    return SupportBackend()


def make_session(backend, steps):
    return Session(ScriptedLLM(steps), kernel=SUPPORT_KERNEL,
                   registry=build_support_registry(backend))


class TestManualData:
    def test_manuals_load_and_section(self):
        manuals = load_manuals()
        assert set(manuals) == {"aquapure_ap100", "smartbrew_sb2"}
        headings = [s["heading"] for s in manuals["aquapure_ap100"]]
        assert "エラーコード一覧" in headings


class TestManualSearchFlow:
    def test_error_code_question_resolved_from_manual(self, backend):
        def answer_from_observation(messages, tools):
            obs = next(m for m in reversed(messages) if m.role == "tool")
            content = str(obs.content)
            assert "E03" in content or "$h" in content
            return ("E03はフィルターカートリッジの寿命のお知らせです。新しいカートリッジ"
                    "(型番 AP-100C)に交換し、リセットボタンを3秒長押ししてください。")

        session = make_session(backend, [
            ScriptedLLM.call("search_manual", query="E03", product="aquapure_ap100"),
            answer_from_observation,
        ])
        reply = session.send("浄水器のAP-100にE03というエラーが出ています")
        assert "AP-100C" in reply
        assert backend.metrics.get("search:aquapure_ap100") == 1

    def test_large_search_result_handled_and_peeked(self, backend):
        session = make_session(backend, [
            ScriptedLLM.call("search_manual", query="エラー 交換 手順 リセット"),
            ScriptedLLM.call("peek", handle="$h1", query="E03"),
            "説明書のとおりカートリッジを交換してください。",
        ])
        session.send("エラーの対処法をぜんぶ教えて")
        obs = [m.content for m in session.conversation if m.role == "tool"]
        assert "$h1" in obs[0]                      # 200tk 閾値超 → ハンドル化 (I7)
        assert "E03" in obs[1]                      # peek で該当行だけ取得
        assert "カートリッジ" in obs[1]

    def test_unknown_product_gets_honest_miss(self, backend):
        session = make_session(backend, [
            ScriptedLLM.call("search_manual", query="ZZ99エラー zz99"),
            "該当する記載が見つかりませんでした。型番をご確認いただけますか?",
        ])
        reply = session.send("ZZ99というエラーが出ます")
        obs = next(m.content for m in session.conversation if m.role == "tool")
        assert "No manual sections matched" in str(obs)
        assert "見つかりません" in reply


class TestEscalationFlow:
    def test_require_spec_gate_then_invalid_email_then_success(self, backend):
        """escalate_to_human is require_spec: the runtime forces a spec review,
        then a bad email raises, then the corrected call goes through."""
        session = make_session(backend, [
            ScriptedLLM.call("escalate_to_human", email="taro@example.com",
                             phone="090-0000-1111", problem_summary="E07が再発する"),
            ScriptedLLM.call("escalate_to_human", email="taro-example.com",
                             phone="090-0000-1111", problem_summary="E07が再発する"),
            ScriptedLLM.call("escalate_to_human", email="taro@example.com",
                             phone="090-0000-1111", problem_summary="E07が再発する"),
            "担当者へ引き継ぎました。チケット番号は TCK-0001 です。",
        ])
        session.send("E07が何度も出ます。人間の方に代わってください。メールはtaro@example.com、電話は090-0000-1111です")
        obs = [str(m.content) for m in session.conversation if m.role == "tool"]
        assert "requires its full spec" in obs[0]     # require_spec gate (§6)
        assert "invalid email" in obs[1]              # handler error → observation
        assert "TCK-0001" in obs[2]

        ticket = backend.tickets[0]
        assert ticket["email"] == "taro@example.com"
        assert ticket["phone"] == "090-0000-1111"
        # 会話全文がチケットに添付される
        assert any("E07が何度も出ます" in t["text"] for t in ticket["transcript"])
        assert backend.metrics["escalation"] == 1


class TestChartCards:
    def test_chart_card_rendered_open_by_default(self, backend):
        session = make_session(backend, [
            ScriptedLLM.call("render_chart", title="月別問い合わせ件数",
                             chart_type="bar",
                             data={"labels": ["4月", "5月", "6月"], "values": [12, 18, 9]}),
            "先月比の問い合わせ件数をグラフにしました。",
        ])
        reply = session.send("問い合わせ件数の推移をグラフで見せて")
        assert "グラフ" in reply
        [card] = backend.charts
        assert card["open"] is True            # カードは既定で開いた状態
        assert card["chart_type"] == "bar"
        assert card["data"]["values"] == [12, 18, 9]

    def test_invalid_chart_type_self_repairs(self, backend):
        session = make_session(backend, [
            ScriptedLLM.call("render_chart", title="t", chart_type="donut", data={}),
            ScriptedLLM.call("render_chart", title="t", chart_type="pie", data={}),
            "円グラフで表示しました。",
        ])
        session.send("ドーナツグラフにして")
        obs = [str(m.content) for m in session.conversation if m.role == "tool"]
        assert "Validation error" in obs[0]
        assert backend.charts[0]["chart_type"] == "pie"


class TestCandidateDiscovery:
    def test_support_tools_surface_as_candidates(self, backend):
        def check(messages, tools):
            names = [t["function"]["name"] for t in tools]
            assert "search_manual" in names, f"layer-2 candidates missing search_manual: {names}"
            return "説明書を確認しますね。"

        session = make_session(backend, [check])
        session.send("説明書のエラー対処の手順を調べてほしい")
