# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Premium Interactive Web UI for Data Cleaning Environment.
Includes WebSocket-powered terminal, live state dashboard, and tool results viewer.
"""

WEB_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenEnv | Data Quality Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --panel: #1e293b;
            --accent: #38bdf8;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Navbar */
        nav {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            padding: 0.75rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
        }
        .logo { font-weight: 700; font-size: 1.25rem; color: var(--accent); }
        .status-badge {
            font-size: 0.75rem;
            padding: 0.25rem 0.75rem;
            border-radius: 99px;
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        /* Main Layout */
        main {
            display: grid;
            grid-template-columns: 1fr 400px;
            flex: 1;
            gap: 1.5rem;
            padding: 1.5rem;
            overflow: hidden;
        }

        /* Panels */
        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .panel-header {
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.02);
        }
        .panel-title { font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }

        /* Dataset View */
        .viewer {
            flex: 1;
            overflow: auto;
            padding: 1rem;
            font-family: 'Fira Code', monospace;
            font-size: 0.85rem;
            white-space: pre;
            background: #020617;
        }
        .table-container { width: 100%; border-collapse: collapse; }

        /* Side Panel */
        .side-panel { display: flex; flex-direction: column; gap: 1.5rem; }
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            padding: 1rem;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            padding: 0.75rem;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value { font-size: 1.25rem; font-weight: 700; color: var(--accent); }
        .stat-label { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.25rem; }

        /* Terminal */
        .terminal {
            flex: 1;
            background: #000;
            padding: 1rem;
            font-family: 'Fira Code', monospace;
            font-size: 0.8rem;
            overflow-y: auto;
            color: #d1d5db;
        }
        .terminal-line { margin-bottom: 0.5rem; }
        .terminal-prompt { color: var(--accent); margin-right: 0.5rem; }
        .terminal-error { color: var(--error); }
        .terminal-success { color: var(--success); }

        /* Controls */
        .controls {
            padding: 1rem;
            border-top: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        select, button {
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.6rem;
            border-radius: 6px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        button:hover { background: var(--border); }
        .btn-primary { background: var(--accent); color: var(--bg); font-weight: 600; border: none; }
        .btn-primary:hover { background: #7dd3fc; }

        /* Tool Result Utility */
        .tool-result-popup {
            background: #1e293b;
            border: 1px solid var(--accent);
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
            font-size: 0.8rem;
        }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
    </style>
</head>
<body>
    <nav>
        <div class="logo">OpenEnv <span style="font-weight: 300; opacity: 0.6;">DataCleaning</span></div>
        <div id="connection-status" class="status-badge">Connecting...</div>
    </nav>

    <main>
        <!-- Center Panel: Terminal & Visualization -->
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">Agent Observation</div>
                <div id="step-counter" style="font-size: 0.8rem;">Step: 0/15</div>
            </div>
            <div id="dataset-viewer" class="viewer">Loading dataset...</div>
            <div id="terminal" class="terminal"></div>
        </div>

        <!-- Right Panel: Controls & State -->
        <div class="side-panel">
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Environment State</div>
                </div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div id="stat-reward" class="stat-value">0.00</div>
                        <div class="stat-label">Total Reward</div>
                    </div>
                    <div class="stat-card">
                        <div id="stat-tasks" class="stat-value">0%</div>
                        <div class="stat-label">Progress</div>
                    </div>
                </div>
                <div style="padding: 1rem; border-top: 1px solid var(--border);">
                    <div style="font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem;">Current Task</div>
                    <div id="task-description" style="font-size: 0.8rem; line-height: 1.5; color: var(--text-muted);">
                        Select a task to begin.
                    </div>
                </div>
                <div class="controls">
                    <select id="task-select">
                        <option value="task_1_identify">Task 1: Identify Errors</option>
                        <option value="task_2_classify">Task 2: Classify Errors</option>
                        <option value="task_3_fix">Task 3: Correct Data</option>
                        <option value="task_4_insight">Task 4: Quality Insights</option>
                    </select>
                    <button class="btn-primary" onclick="resetEnv()">Reset Environment</button>
                </div>
            </div>

            <div class="panel" style="flex: 1;">
                <div class="panel-header">
                    <div class="panel-title">Action History</div>
                </div>
                <div id="action-history" style="overflow-y: auto; padding: 1rem;">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">No actions taken yet.</div>
                </div>
            </div>
        </div>
    </main>

    <script>
        const terminal = document.getElementById('terminal');
        const datasetViewer = document.getElementById('dataset-viewer');
        const statusBadge = document.getElementById('connection-status');
        const taskSelect = document.getElementById('task-select');
        
        let ws;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        function connect() {
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                statusBadge.innerText = 'Connected';
                statusBadge.style.color = '#10b981';
                log('System', 'Connected to environment server.');
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'observation') {
                    updateUI(msg.data.observation);
                } else if (msg.type === 'state') {
                    updateState(msg.data);
                }
            };

            ws.onclose = () => {
                statusBadge.innerText = 'Disconnected';
                statusBadge.style.color = '#ef4444';
                log('System', 'Disconnected. Attempting reconnect...', 'error');
                setTimeout(connect, 3000);
            };
        }

        function log(source, text, type = 'info') {
            const line = document.createElement('div');
            line.className = 'terminal-line';
            const prompt = document.createElement('span');
            prompt.className = 'terminal-prompt';
            prompt.innerText = `[${new Date().toLocaleTimeString()}] ${source}>`;
            
            const content = document.createElement('span');
            content.innerText = text;
            if (type === 'error') content.className = 'terminal-error';
            if (type === 'success') content.className = 'terminal-success';

            line.appendChild(prompt);
            line.appendChild(content);
            terminal.appendChild(line);
            terminal.scrollTop = terminal.scrollHeight;
        }

        function updateUI(obs) {
            datasetViewer.innerText = obs.dataset_text;
            document.getElementById('task-description').innerText = obs.task_description;
            document.getElementById('step-counter').innerText = `Step: ${obs.step_number}/${obs.max_steps}`;
            
            if (obs.feedback) {
                log('Env', obs.feedback, obs.reward > 0 ? 'success' : 'info');
            }

            if (obs.tool_output) {
                const toolBlock = document.createElement('div');
                toolBlock.className = 'tool-result-popup';
                toolBlock.innerHTML = `<strong>Tool Result:</strong><pre style="margin-top: 0.5rem; color: #38bdf8;">${JSON.stringify(obs.tool_output, null, 2)}</pre>`;
                terminal.appendChild(toolBlock);
                terminal.scrollTop = terminal.scrollHeight;
            }

            document.getElementById('stat-reward').innerText = obs.reward.toFixed(3);
        }

        function updateState(state) {
            document.getElementById('stat-reward').innerText = state.cumulative_reward.toFixed(3);
            const progress = (state.step_count / 15 * 100).toFixed(0);
            document.getElementById('stat-tasks').innerText = `${progress}%`;
        }

        function resetEnv() {
            const task = taskSelect.value;
            ws.send(json.dumps({
                type: 'reset',
                data: { task_name: task }
            }));
            terminal.innerHTML = '';
            log('Client', `Resetting to ${task}...`);
        }

        connect();
    </script>
</body>
</html>
"""
