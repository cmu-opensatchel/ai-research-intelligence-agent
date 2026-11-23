import html
import re
import subprocess
import sys

import streamlit as st

st.set_page_config(page_title="AI Research Intelligence Agent", layout="wide")

st.markdown("""
<style>
.main .block-container {padding-top: 2rem;}
.output-box {background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 1rem; font-family: monospace; font-size: 0.85rem; overflow-x: auto; white-space: pre-wrap; color: #222;}
.candidate-card {background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
.candidate-header {font-size: 1.1rem; font-weight: bold; color: #1a1a2e; margin-bottom: 0.5rem;}
.candidate-meta {color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;}
.source-badge {background: #e8f0f7; color: #1a1a2e; padding: 3px 8px; border-radius: 4px; font-size: 0.85rem; display: inline-block; margin-right: 5px;}
.score-badge {background: #27ae60; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85rem; margin-right: 5px;}
.assessment-badge {background: #3498db; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85rem; margin-right: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("🔍 AI Research Intelligence Agent")
st.markdown("### Multi-Agent System for Monitoring, Verifying, and Synthesizing Emerging AI Developments")

st.sidebar.header("📊 Analysis Controls")
topics = st.sidebar.text_input("Topics", value="agentic AI")
window = st.sidebar.slider("Window (days)", 1, 90, 7)
run_llm = st.sidebar.checkbox("Enable LLM Verification", value=False)

if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False

st.markdown("### 🎯 Search Parameters")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Topics:** `{topics}`")
with col2:
    st.markdown(f"**Window:** `{window}` days")

st.markdown("---")


def parse_output(output_lines):
    """Extract artifact path, termination reason and ranked candidates from CLI stdout."""
    artifact_path = None
    termination = None
    candidates = []
    current = None
    field = None  # which field wrapped continuation lines belong to

    box_chars = set("┏┓┗┛┃━┡┩└┘│─┳╇┻╈┼├┤┌┐┬┴")
    field_keys = {
        "assessment": "assessment",
        "score": "score",
        "source": "source",
        "link": "link",
        "url": "link",
    }
    # Other "Label:" lines we recognise only so they end a wrapped field.
    other_keys = {"summary", "why it matters", "rationale", "verification",
                  "evidence", "date", "published", "topics", "reporting window"}

    for line in output_lines:
        stripped = line.strip()

        # Lines printed after the panel
        m = re.search(r"Run artifact saved to:\s*(.+)", stripped)
        if m:
            artifact_path = m.group(1).strip()
            field = None
            continue
        m = re.search(r"Termination reason:\s*(.+)", stripped)
        if m:
            termination = m.group(1).strip()
            field = None
            continue

        # The Rich "Candidate Rankings" table truncates titles with "…"; ignore it.
        if any(ch in box_chars for ch in stripped):
            continue

        # Panel border lines like "+-----+" end the current field.
        if re.match(r"^\+[-=+ ]*\+?$", stripped):
            field = None
            continue

        # Remove the panel's "| ... |" frame and any markdown bold markers.
        text = stripped
        if text.startswith("|"):
            text = text[1:]
        if text.endswith("|"):
            text = text[:-1]
        text = text.replace("**", "").strip()

        if not text:
            field = None
            continue

        # "## 1. Title..." starts a new candidate.
        m = re.match(r"##\s+(\d+)\.\s+(.*)", text)
        if m:
            if current:
                candidates.append(current)
            current = {
                "rank": int(m.group(1)),
                "title": m.group(2).strip(),
                "assessment": "",
                "score": "",
                "source": "",
                "link": "",
            }
            field = "title"
            continue

        if current is None:
            continue

        # "Label: value" lines
        m = re.match(r"^([A-Za-z][A-Za-z ]{0,30}):\s*(.*)$", text)
        if m:
            label = m.group(1).strip().lower()
            if label in field_keys:
                field = field_keys[label]
                current[field] = m.group(2).strip()
                continue
            if label in other_keys:
                field = None
                continue

        # Any other "## heading" or "# heading" ends the current candidate's fields.
        if text.startswith("#"):
            field = None
            continue

        # Otherwise this is a wrapped continuation of the current field.
        if field:
            current[field] = (current[field] + " " + text).strip()

    if current:
        candidates.append(current)
    return artifact_path, termination, candidates


if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
    st.session_state.is_running = True
    st.session_state.last_results = None  # clear any stale result/error

    try:
        with st.spinner("Running analysis... please wait"):
            cmd = [
                sys.executable, "-m", "ai_research_agent.cli", "run",
                "--topics", topics,
                "--window", str(window),
            ]
            if run_llm:
                cmd.append("--use-llm-verification")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            output_lines = result.stdout.splitlines() if result.stdout else []
            error_lines = result.stderr.splitlines() if result.stderr else []

            artifact_path, termination, candidates = parse_output(output_lines)

            st.session_state.last_results = {
                "stdout": output_lines,
                "stderr": error_lines,
                "returncode": result.returncode,
                "artifact_path": artifact_path,
                "termination": termination,
                "topics": topics,
                "window": window,
                "llm": run_llm,
                "candidates": candidates,
                "raw_lines_count": len(output_lines),
                "first_10_lines": output_lines[:10],
            }
    except subprocess.TimeoutExpired:
        st.session_state.last_results = {"error": "Timed out after 10 minutes"}
    except Exception as e:
        st.session_state.last_results = {"error": f"{type(e).__name__}: {e}"}
    finally:
        st.session_state.is_running = False

# Display results
if st.session_state.get("last_results"):
    results = st.session_state.last_results

    if "error" in results:
        st.error(f"❌ {results['error']}")
    else:
        st.markdown("## 📊 ROUNDUP REPORT")
        st.markdown("---")

        failed = results.get("returncode", 0) != 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Analysis Status**")
            if failed:
                st.markdown(f"❌ **Failed (exit code {results['returncode']})**")
            else:
                st.markdown("✅ **Completed Successfully**")
        with col2:
            if results.get("artifact_path"):
                st.markdown(f"**Artifact**\n`{results['artifact_path']}`")
        with col3:
            if results.get("termination"):
                st.markdown(f"**Status**\n{results['termination']}")

        if failed and results.get("stderr"):
            with st.expander("⚠️ Error output (stderr)", expanded=True):
                st.code("\n".join(results["stderr"]))

        st.markdown("### 🏆 CANDIDATE RANKINGS")
        st.markdown("---")

        candidates = results.get("candidates", [])
        if candidates:
            for c in candidates[:10]:
                rank = c.get("rank", 0)
                title = html.escape(str(c.get("title", "")))
                assessment = html.escape(str(c.get("assessment", "")))
                score = html.escape(str(c.get("score", "")))
                source_raw = str(c.get("source", ""))
                source = html.escape(source_raw[:50] + ("..." if len(source_raw) > 50 else ""))

                badges = ""
                if score:
                    badges += f'<span class="score-badge">{score}</span>'
                if assessment:
                    badges += f'<span class="assessment-badge">{assessment}</span>'
                if source:
                    badges += f'<span class="source-badge">{source}</span>'

                # Single-line HTML: indented HTML would render as a code block in Markdown.
                card = (
                    '<div class="candidate-card">'
                    f'<div class="candidate-header">#{rank}. {title}</div>'
                    f'<div class="candidate-meta">{badges}</div>'
                    "</div>"
                )
                st.markdown(card, unsafe_allow_html=True)
        else:
            st.info("No candidates found in output")
            with st.expander("🔍 Debug: Raw Output Analysis", expanded=True):
                st.markdown(f"**Total output lines:** {results.get('raw_lines_count', 0)}")
                st.markdown("**First 10 lines:**")
                for i, line in enumerate(results.get("first_10_lines", [])):
                    st.text(f"{i + 1}: {line!r}")

        st.markdown("---")

        with st.expander("📋 View Detailed Output Log", expanded=False):
            stdout_text = "\n".join(results.get("stdout", []))
            if stdout_text:
                st.markdown(
                    f'<div class="output-box">{html.escape(stdout_text)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No output captured")

        st.markdown("---")
        st.download_button(
            "📥 Download Full Output",
            data="\n".join(results.get("stdout", [])),
            file_name="analysis_output.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if st.button("🔄 Run Another Analysis"):
        st.session_state.last_results = None
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ System Status")


@st.cache_data(ttl=30)
def _ollama_status():
    """Check (not assume) whether the local Qwen3-8B model is reachable via Ollama."""
    try:
        from ai_research_agent.config import AgentConfig
        from ai_research_agent.llm import OllamaClient

        cfg = AgentConfig()
        return OllamaClient(cfg).is_available(), cfg.ollama_model
    except Exception:
        return False, ""


_llm_ready, _llm_model = _ollama_status()
st.sidebar.markdown("✅ Core: heuristic verification + bounded Tree of Thought")
if _llm_ready:
    st.sidebar.markdown(f"✅ Local LLM available: {_llm_model}")
else:
    st.sidebar.markdown("⚪ Local LLM (Ollama): not available; LLM verification will fall back to heuristics")
