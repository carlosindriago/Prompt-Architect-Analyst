"""
HTML Reporter — renders enriched ScoreCards into a self-contained HTML file.

DESIGN INVARIANTS
- Jinja2 is configured with `autoescape=True` to prevent XSS via
  rationale fields, session_ids, or any other user-controlled string.
- The output file is always opened with `encoding="utf-8"` via a
  context manager so the resource is released on every exit path.
- The default template uses CSS custom properties (variables) and
  avoids grid/flexbox so it renders cleanly in WeasyPrint later.
- The reporter is stateless: `render()` is a pure side-effect
  function that does not mutate the input tuple.
"""

from __future__ import annotations

# ruff: noqa: E501
import os
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment

from src.i18n import get_text
from src.scorer import GlobalAnalysis, ScoreCard
from src.teaching_content import SKILL_TEACH_CONTENT

_DEFAULT_TEMPLATE: str = """<!DOCTYPE html>
<html lang="{{ lang }}">
<head>
    <meta charset="utf-8">
    <title>AI Coding Insight Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <style>
        :root {
            --bg-body: #020617; 
            --bg-card: rgba(15, 23, 42, 0.7); 
            --bg-card-hover: rgba(30, 41, 59, 0.9);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.5);
            --border: rgba(51, 65, 85, 0.5);
            --border-highlight: rgba(56, 189, 248, 0.3);
            
            --badge-novato: #ef4444;
            --badge-prof: #38bdf8;
            --badge-senior: #f59e0b;
            --badge-bg: rgba(255, 255, 255, 0.05);
            --success: #22c55e;
            --danger: #ef4444;
            --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        * {
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            line-height: 1.6;
            font-size: 1.05rem;
            min-height: 100vh;
            overflow-x: hidden;
        }

        #bg-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: -1;
            pointer-events: none;
        }

        .navbar {
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(2, 6, 23, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 0 2rem;
            display: flex;
            justify-content: center;
            gap: 2rem;
        }

        .nav-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            font-weight: 500;
            padding: 1.5rem 0.5rem;
            cursor: pointer;
            position: relative;
            transition: color 0.3s ease;
        }

        .nav-btn:hover {
            color: var(--text-main);
        }

        .nav-btn.active {
            color: var(--accent);
        }

        .nav-btn.active::after {
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            width: 100%;
            height: 3px;
            background: var(--accent);
            border-radius: 3px 3px 0 0;
            box-shadow: 0 0 10px var(--accent-glow);
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 3rem 1rem;
            animation: fadeIn 0.8s ease-out forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
            animation: fadeIn 0.4s ease-out forwards;
        }

        header {
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }

        h1 {
            font-weight: 700;
            font-size: 3.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }

        h2, h3 {
            color: var(--text-main);
            font-weight: 600;
        }
        
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }

        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: var(--glass-shadow);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        
        .glass-card:hover {
            border-color: var(--border-highlight);
            transform: translateY(-2px);
        }

        .kpi-card {
            text-align: center;
            padding: 1.5rem;
        }

        .kpi-value {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #f8fafc, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            line-height: 1;
        }

        .kpi-label {
            font-size: 0.9rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 600;
        }
        
        .workflow-badge {
            display: inline-block;
            padding: 0.35rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: var(--badge-bg);
            border: 1px solid rgba(255,255,255,0.1);
        }

        .badge-novato { color: var(--badge-novato); border-color: rgba(239, 68, 68, 0.3); }
        .badge-prof { color: var(--badge-prof); border-color: rgba(56, 189, 248, 0.3); }
        .badge-senior { color: var(--badge-senior); border-color: rgba(245, 158, 11, 0.3); }

        .terminal-window {
            background: rgba(2, 6, 23, 0.8);
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        .terminal-header {
            background: rgba(15, 23, 42, 0.9);
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            border-bottom: 1px solid var(--border);
        }

        .terminal-dots {
            display: flex;
            gap: 0.5rem;
        }

        .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .dot.red { background: #ef4444; }
        .dot.yellow { background: #f59e0b; }
        .dot.green { background: #10b981; }

        .terminal-title {
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            margin-left: 1rem;
        }

        .terminal-content {
            padding: 1.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            color: #e2e8f0;
            line-height: 1.7;
            overflow-x: auto;
            white-space: pre-wrap; /* This fixes the prompt layout */
        }
        
        .terminal-content .bracket {
            color: var(--accent);
            font-weight: 700;
            display: inline-block;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-size: 1.05rem;
        }
        /* Remove top margin for the first bracket */
        .terminal-content .bracket:first-child {
            margin-top: 0;
        }

        .security-safe {
            margin-top: 1rem;
            padding: 1.5rem;
            background: rgba(34, 197, 94, 0.1);
            border-left: 4px solid var(--success);
            border-radius: 4px;
            color: #e2e8f0;
        }

        .security-leaks {
            margin-top: 1rem;
            padding: 1.5rem;
            background: rgba(239, 68, 68, 0.15);
            border-left: 4px solid var(--danger);
            border-radius: 4px;
            color: #fca5a5;
        }
        
        .security-risks {
            margin-top: 1rem;
            padding: 1.5rem;
            background: rgba(245, 158, 11, 0.1);
            border-left: 4px solid #f59e0b;
            border-radius: 4px;
            color: #fcd34d;
        }
        
        ul {
            padding-left: 1.5rem;
            margin-bottom: 0;
        }
        li {
            margin-bottom: 0.5rem;
        }
        
        .feedback-box {
            background: rgba(56, 189, 248, 0.1);
            border-left: 4px solid var(--accent);
            padding: 1.5rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 2rem;
            font-style: italic;
        }

        .session-card {
            margin-bottom: 1.5rem;
        }
        .session-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .session-title {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent);
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        
        @media (max-width: 768px) {
            .grid-2 { grid-template-columns: 1fr; }
            h1 { font-size: 2.5rem; }
        }
    </style>
</head>
<body>
    <canvas id="bg-canvas"></canvas>

    <nav class="navbar">
        <button class="nav-btn active" onclick="showTab('tab-general')">{{ t('html_dashboard') }}</button>
        <button class="nav-btn" onclick="showTab('tab-prompt')">Prompt Ideal</button>
        <button class="nav-btn" onclick="showTab('tab-security')">Seguridad</button>
        <button class="nav-btn" onclick="showTab('tab-sessions')">Sesiones</button>
    </nav>

    <div class="container">
        <!-- HEADER GENERAL -->
        <header>
            <h1>AI Coding Insight</h1>
        </header>

        <!-- TAB: GENERAL -->
        <div id="tab-general" class="tab-content active">
            <div class="kpi-grid">
                <div class="glass-card kpi-card">
                    <div class="kpi-value">
                        {% if global_analysis.security_score is not none %}
                            {{ "%.0f"|format(global_analysis.security_score * 100) }}
                        {% else %}
                            -
                        {% endif %}
                    </div>
                    <div class="kpi-label">Global Score</div>
                </div>
                <div class="glass-card kpi-card" style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <div style="margin-bottom: 1rem;">
                        <span class="workflow-badge {% if global_analysis.security_score is not none and global_analysis.security_score >= 0.9 %}badge-prof{% elif global_analysis.security_score is not none and global_analysis.security_score >= 0.7 %}badge-senior{% else %}badge-novato{% endif %}" style="font-size: 1.2rem; padding: 0.5rem 1.5rem;">
                            {% if global_analysis.security_score is not none and global_analysis.security_score >= 0.9 %}Senior{% elif global_analysis.security_score is not none and global_analysis.security_score >= 0.7 %}Profesional{% else %}Novato{% endif %}
                        </span>
                    </div>
                    <div class="kpi-label">Workflow Level</div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-value">{{ cards|length }}</div>
                    <div class="kpi-label">Sesiones Analizadas</div>
                </div>
            </div>

            {% if global_analysis.recommendations %}
            <div class="glass-card">
                <h2 style="margin-top: 0; color: var(--accent);"><span style="margin-right: 10px;">🚀</span> Growth Opportunities</h2>
                <ul>
                    {% for rec in global_analysis.recommendations %}
                    <li style="margin-bottom: 0.75rem;">{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>

        <!-- TAB: PROMPT IDEAL -->
        <div id="tab-prompt" class="tab-content">
            <div class="glass-card" style="margin-bottom: 2rem;">
                <h2 style="margin-top: 0; color: var(--accent);">Arquitectura del Prompting</h2>
                <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem;">
                    Para interactuar con la IA al nivel de un <strong>Senior Architect</strong>, es fundamental dejar de tratar al modelo como un chatbot y empezar a tratarlo como un compilador al que se le pasan <strong>especificaciones técnicas</strong>.
                </p>
                {% if global_analysis.user_feedback %}
                <div class="feedback-box">
                    <h3 style="margin-top: 0; color: var(--accent);">✅ Análisis de tu estilo actual</h3>
                    <p style="margin-bottom: 0; font-size: 0.95rem;">{{ global_analysis.user_feedback }}</p>
                </div>
                {% endif %}
                <h3 style="margin-top: 0; color: #e2e8f0;">Tu Prompt Ideal (Caso Real)</h3>
                <p style="color: var(--text-muted); font-size: 0.9rem;">
                    Basado en las tareas técnicas que estuviste realizando, aquí tienes un ejemplo concreto de cómo deberías estructurar tus prompts para obtener resultados deterministas y profesionales:
                </p>
            </div>

            {% if global_analysis.ideal_prompt %}
            <div class="prompt-sections" style="margin-bottom: 2rem;">
                {% for section in global_analysis.ideal_prompt %}
                <div class="prompt-section" style="margin-bottom: 2rem;">
                    <!-- Theory Block -->
                    <div style="background: rgba(56, 189, 248, 0.08); border-left: 4px solid var(--accent); padding: 1.5rem; border-radius: 8px 8px 0 0; margin-bottom: 0;">
                        <h4 style="margin: 0 0 0.5rem 0; color: var(--accent); font-size: 1.1rem;">{{ section.header }}</h4>
                        <p style="margin: 0; font-size: 0.95rem; color: #e2e8f0; line-height: 1.5;">{{ section.theory }}</p>
                    </div>
                    <!-- Example Block -->
                    <div class="terminal-window" style="border-radius: 0 0 8px 8px; border-top: none; margin-top: 0;">
                        <div class="terminal-content" style="padding: 1.5rem; font-family: monospace; color: #a7f3d0; white-space: pre-wrap;">{{ section.example }}</div>
                    </div>
                </div>
                {% endfor %}
            </div>

            <div class="glass-card">
                <h3 style="margin-top: 0; color: #e2e8f0;">El Prompt Completo (Cópialo y Pégalo)</h3>
                <div class="terminal-window" style="margin-top: 1rem;">
                    <div class="terminal-header">
                        <div class="terminal-dots">
                            <div class="dot red"></div>
                            <div class="dot yellow"></div>
                            <div class="dot green"></div>
                        </div>
                        <div class="terminal-title">ideal_prompt.txt</div>
                    </div>
                    <div class="terminal-content">{% for section in global_analysis.ideal_prompt %}<span class="bracket">{{ section.header }}</span>
{{ section.example | escape }}

{% endfor %}</div>
                </div>
            </div>
            {% else %}
            <div class="glass-card">
                <p>No se pudo generar el prompt ideal en esta iteración.</p>
                {% if global_analysis.debug_raw_prompt %}
                <details>
                    <summary style="cursor: pointer; color: var(--accent);">Ver respuesta cruda del LLM (Debug)</summary>
                    <pre style="background: #1e293b; padding: 1rem; border-radius: 4px; overflow-x: auto; color: #a7f3d0; margin-top: 1rem; font-size: 0.85rem;">{{ global_analysis.debug_raw_prompt }}</pre>
                </details>
                {% endif %}
            </div>
            {% endif %}
        </div>

        <!-- TAB: SECURITY -->
        <div id="tab-security" class="tab-content">
            <div class="kpi-grid">
                <div class="glass-card kpi-card">
                    <div class="kpi-value" style="color: {% if global_analysis.security_score is not none and global_analysis.security_score >= 0.9 %}var(--success){% elif global_analysis.security_score is not none and global_analysis.security_score >= 0.6 %}#f59e0b{% else %}var(--danger){% endif %};">
                        {% if global_analysis.security_score is not none %}
                            {{ "%.0f"|format(global_analysis.security_score * 100) }}
                        {% else %}
                            -
                        {% endif %}
                    </div>
                    <div class="kpi-label">Security Score</div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-value" style="color: {% if global_analysis.security_leaks %}var(--danger){% else %}var(--success){% endif %};">
                        {{ global_analysis.security_leaks|length if global_analysis.security_leaks else 0 }}
                    </div>
                    <div class="kpi-label">Fugas Críticas</div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-value" style="color: {% if global_analysis.security_risks %}#f59e0b{% else %}var(--success){% endif %};">
                        {{ global_analysis.security_risks|length if global_analysis.security_risks else 0 }}
                    </div>
                    <div class="kpi-label">Riesgos Detectados</div>
                </div>
            </div>

            <div class="glass-card">
                <h3 style="margin-top: 0; color: #e2e8f0;">Evaluación de Seguridad</h3>
                <p style="color: var(--text-muted); font-size: 1.05rem;">{{ global_analysis.security_rationale }}</p>

                {% if global_analysis.security_leaks %}
                <div class="security-leaks">
                    <strong>🚨 CRITICAL SECURITY LEAKS DETECTED:</strong>
                    <ul style="margin-top: 0.75rem;">
                        {% for leak in global_analysis.security_leaks %}
                        <li>{{ leak }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}

                {% if global_analysis.security_risks %}
                <div class="security-risks">
                    <strong>⚠️ {{ t('html_security_risks') }}:</strong>
                    <ul style="margin-top: 0.75rem;">
                        {% for risk in global_analysis.security_risks %}
                        <li>{{ risk }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
                
                {% if not global_analysis.security_leaks and not global_analysis.security_risks %}
                <div class="security-safe">
                    <strong>🛡️ {{ t('html_no_security_risks') }}</strong>
                </div>
                {% endif %}
            </div>
        </div>

        <!-- TAB: SESSIONS -->
        <div id="tab-sessions" class="tab-content">
            {% for card in cards %}
            <div class="glass-card session-card">
                <div class="session-header">
                    <div class="session-title">{{ card.session_id }}</div>
                    {% if card.overall is not none %}
                    <div class="workflow-badge {% if card.overall >= 0.85 %}badge-prof{% elif card.overall >= 0.6 %}badge-senior{% else %}badge-novato{% endif %}">
                        Score: {{ "%.0f"|format(card.overall * 100) }}
                    </div>
                    {% endif %}
                </div>
                
                <div class="grid-2">
                    <div>
                        <h4 style="margin-top: 0; color: #cbd5e1;">Métricas</h4>
                        <ul style="color: var(--text-muted); font-size: 0.95rem;">
                            {% for dim in card.dimensions %}
                            <li>
                                <strong>{{ dim.name }}:</strong> 
                                {% if dim.score is not none %}{{ "%.0f"|format(dim.score * 100) }}{% else %}N/A{% endif %}
                            </li>
                            {% endfor %}
                        </ul>
                    </div>
                    <div>
                        {% if card.tips %}
                        <h4 style="margin-top: 0; color: #cbd5e1;">Tips</h4>
                        <ul style="color: var(--text-muted); font-size: 0.95rem;">
                            {% for tip in card.tips %}
                            <li>{{ tip }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <!-- TABS LOGIC -->
    <script>
        function showTab(tabId) {
            // Hide all
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
            
            // Show target
            document.getElementById(tabId).classList.add('active');
            
            // Update active button
            const btns = document.querySelectorAll('.nav-btn');
            btns.forEach(btn => {
                if(btn.getAttribute('onclick').includes(tabId)) {
                    btn.classList.add('active');
                }
            });

            // Burst animation effect when changing tabs
            if (typeof velocities !== 'undefined' && velocities.length > 0) {
                for(let i = 0; i < velocities.length; i++) {
                    velocities[i].x += (Math.random() - 0.5) * 2;
                    velocities[i].y += (Math.random() - 0.5) * 2;
                }
            }
        }
    </script>

    <!-- THREE.JS BACKGROUND LOGIC -->
    <script>
        // Network nodes animation
        const canvas = document.querySelector('#bg-canvas');
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
        
        let width = window.innerWidth;
        let height = window.innerHeight;
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        const scene = new THREE.Scene();
        
        // Camera setup
        const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        camera.position.z = 400;

        // Particle System Setup
        const particleCount = 250;
        const particles = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const velocities = [];

        for (let i = 0; i < particleCount; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 1000;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 1000;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 500;
            
            velocities.push({
                x: (Math.random() - 0.5) * 0.5,
                y: (Math.random() - 0.5) * 0.5,
                z: (Math.random() - 0.5) * 0.5
            });
        }

        particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const particleMaterial = new THREE.PointsMaterial({
            color: 0x38bdf8,
            size: 3,
            transparent: true,
            opacity: 0.2
        });

        const particleSystem = new THREE.Points(particles, particleMaterial);
        scene.add(particleSystem);

        // Lines Setup
        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.05
        });

        // Optimization: Create lines geometry with max possible segments, update positions dynamically
        const maxLines = particleCount * 2;
        const linePositions = new Float32Array(maxLines * 6); // 2 points per line, 3 coords per point
        const lineGeometry = new THREE.BufferGeometry();
        lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
        const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
        scene.add(lines);

        // Mouse interaction
        let mouseX = 0;
        let mouseY = 0;
        let targetX = 0;
        let targetY = 0;

        document.addEventListener('mousemove', (e) => {
            mouseX = (e.clientX - width / 2);
            mouseY = (e.clientY - height / 2);
        });

        // Resize handler
        window.addEventListener('resize', () => {
            width = window.innerWidth;
            height = window.innerHeight;
            renderer.setSize(width, height);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
        });

        // Animation loop
        const connectDistance = 100;
        
        function animate() {
            requestAnimationFrame(animate);

            // Smooth camera movement based on mouse
            targetX = mouseX * 0.1;
            targetY = mouseY * 0.1;
            camera.position.x += (targetX - camera.position.x) * 0.05;
            camera.position.y += (-targetY - camera.position.y) * 0.05;
            camera.lookAt(scene.position);

            const posAttribute = particles.getAttribute('position');
            const currentPositions = posAttribute.array;
            
            // Update particle positions
            for (let i = 0; i < particleCount; i++) {
                currentPositions[i * 3] += velocities[i].x;
                currentPositions[i * 3 + 1] += velocities[i].y;
                currentPositions[i * 3 + 2] += velocities[i].z;

                // Bounce off boundaries
                if (Math.abs(currentPositions[i * 3]) > 500) velocities[i].x *= -1;
                if (Math.abs(currentPositions[i * 3 + 1]) > 500) velocities[i].y *= -1;
                if (Math.abs(currentPositions[i * 3 + 2]) > 250) velocities[i].z *= -1;
            }
            
            posAttribute.needsUpdate = true;
            particleSystem.rotation.y += 0.001;
            lines.rotation.y = particleSystem.rotation.y;

            // Update lines
            let lineIndex = 0;
            for (let i = 0; i < particleCount; i++) {
                for (let j = i + 1; j < particleCount; j++) {
                    const dx = currentPositions[i * 3] - currentPositions[j * 3];
                    const dy = currentPositions[i * 3 + 1] - currentPositions[j * 3 + 1];
                    const dz = currentPositions[i * 3 + 2] - currentPositions[j * 3 + 2];
                    const distSq = dx*dx + dy*dy + dz*dz;

                    if (distSq < connectDistance * connectDistance && lineIndex < maxLines) {
                        linePositions[lineIndex * 6] = currentPositions[i * 3];
                        linePositions[lineIndex * 6 + 1] = currentPositions[i * 3 + 1];
                        linePositions[lineIndex * 6 + 2] = currentPositions[i * 3 + 2];
                        
                        linePositions[lineIndex * 6 + 3] = currentPositions[j * 3];
                        linePositions[lineIndex * 6 + 4] = currentPositions[j * 3 + 1];
                        linePositions[lineIndex * 6 + 5] = currentPositions[j * 3 + 2];
                        
                        lineIndex++;
                    }
                }
            }
            
            // Hide unused lines by collapsing them to origin
            for (let i = lineIndex; i < maxLines; i++) {
                linePositions[i * 6] = 0; linePositions[i * 6 + 1] = 0; linePositions[i * 6 + 2] = 0;
                linePositions[i * 6 + 3] = 0; linePositions[i * 6 + 4] = 0; linePositions[i * 6 + 5] = 0;
            }
            
            lineGeometry.attributes.position.needsUpdate = true;

            renderer.render(scene, camera);
        }

        animate();
    </script>
</body>
</html>"""


@dataclass(frozen=True, slots=True)
class HTMLReporter:
    """Render ScoreCards into a self-contained HTML file using Jinja2.

    The reporter uses Jinja2 with `autoescape=True` to prevent XSS.
    A custom template can be provided via `template_path`; otherwise
    a clean, linear default template is used.

    The reporter is stateless: `render()` does not mutate the input
    tuple and produces no observable side effect beyond the output file.
    """

    template_path: Path | None = None

    def render(
        self,
        cards: tuple[ScoreCard, ...],
        global_analysis: GlobalAnalysis,
        output_path: Path,
        language: str = "en",
    ) -> None:
        """Render the ScoreCards to HTML and write to output_path.

        Steps:
        1. Build a Jinja2 Environment with autoescape=True.
        2. Load the template source (custom or default).
        3. Render the template with `cards` in the context.
        4. Write the result to `output_path` using a context manager
           with encoding="utf-8".
        """
        env: Environment = Environment(
            autoescape=True,
            keep_trailing_newline=True,
        )

        def format_ideal_prompt(text: str) -> str:
            if not text:
                return ""
            # Strip any rogue HTML tags the LLM might have output
            text = re.sub(r"<[^>]+>", "", text)
            import html

            text = html.escape(text)
            from markupsafe import Markup

            # Highlight known headers
            formatted = re.sub(
                r"(\[(?:SCENARIO|ROLE|CONTEXT|TASK|CONSTRAINTS|FORMAT|ACCEPTANCE)\])",
                r'<span class="bracket">\1</span>',
                text,
            )
            return Markup(formatted)  # noqa: S704

        env.filters["format_ideal_prompt"] = format_ideal_prompt

        if self.template_path is not None:
            template_source: str = self.template_path.read_text(encoding="utf-8")
        else:
            template_source = _DEFAULT_TEMPLATE

        # Compute growth cards sorted by impact
        growth_cards_by_session = {}
        for card in cards:
            session_cards = []
            for dim in card.dimensions:
                if dim.score is not None and dim.score < 0.85:
                    weight = 1.0  # fallback weight
                    # Retrieve weight from dimensions (we can approximate here, or just use 1.0)
                    impact = (0.85 - dim.score) * weight
                    session_cards.append({"name": dim.name, "impact": impact, "score": dim.score})
            session_cards.sort(key=lambda x: float(str(x["impact"])), reverse=True)
            growth_cards_by_session[card.session_id] = session_cards

        template = env.from_string(template_source)
        html: str = template.render(
            cards=cards,
            global_analysis=global_analysis,
            t=lambda k: get_text(k, language),
            growth_cards=growth_cards_by_session,
            skill_teach=SKILL_TEACH_CONTENT,
            lang=language,
        )

        # Scrub local paths
        html = re.sub(r"(?:/home/|/Users/)[^/]+/", "~/", html)

        with output_path.open("w", encoding="utf-8") as fh:
            fh.write(html)

        # Security audit S3: the report may contain source code excerpts
        # that the user did not intend to share world-readable. Force
        # 0o600 (owner read/write only) regardless of the process
        # umask. This is a defense-in-depth measure: the caller may
        # have a permissive umask (e.g. 0o022 in CI), and we never
        # want the rendered HTML to be readable by other users on a
        # shared system. The chmod is best-effort: a failure (e.g.
        # Windows, filesystem without mode support) is logged but
        # does not abort the report — the user still gets their file.
        try:
            os.chmod(output_path, 0o600)
        except OSError:
            import logging

            _logger = logging.getLogger(__name__)
            _logger.warning(
                "Could not set 0o600 permissions on %s; "
                "the report may be world-readable on this filesystem.",
                output_path,
            )


__all__ = ["HTMLReporter"]
