"""
SKILL_TEACH corpus for providing actionable, educational growth cards based on low scores.
"""

from typing import Any

SKILL_TEACH_CONTENT: dict[str, dict[str, Any]] = {
    "Direction": {
        "en": {
            "what_it_is": "Providing clear, atomic tasks with constraints and expected outputs.",
            "why_it_matters": "Agents are literal. Ambiguity forces them to guess, leading to hallucinations and re-work.",
            "how_to_improve": "Use a structured prompt: Role, Context, Task, Constraints, Expected Output.",
            "example": "Instead of 'fix this', say 'Fix the NullReferenceException in UserService.login() without changing the method signature.'",
            "good_looks_like": "A single prompt that yields the exact desired code on the first try.",
        },
        "es": {
            "what_it_is": "Proveer tareas claras, atómicas con restricciones y formato de salida esperado.",
            "why_it_matters": "Los agentes son literales. La ambigüedad los obliga a adivinar, llevando a alucinaciones.",
            "how_to_improve": "Usa una estructura: Rol, Contexto, Tarea, Restricciones, Salida Esperada.",
            "example": "En lugar de 'arregla esto', usa 'Arregla la NullReferenceException en UserService.login() sin alterar la firma del método.'",
            "good_looks_like": "Un prompt directo que produce el código exacto a la primera.",
        },
    },
    "Verification": {
        "en": {
            "what_it_is": "Running tests or builds immediately after the agent writes code.",
            "why_it_matters": "Unverified code compounds errors. If the agent makes a mistake in step 1, step 2 will be flawed.",
            "how_to_improve": "After every file edit, ask the agent to run the test suite or linter.",
            "example": "After a refactor, say 'Run pytest on tests/test_user.py to verify'.",
            "good_looks_like": "A workflow where every write operation is followed by a read/verify operation.",
        },
        "es": {
            "what_it_is": "Ejecutar pruebas o compilaciones inmediatamente después de que el agente escribe código.",
            "why_it_matters": "El código sin verificar acumula errores. Si hay un error en el paso 1, el paso 2 fallará.",
            "how_to_improve": "Después de cada edición, pide al agente correr el linter o los tests.",
            "example": "Después de un refactor, di 'Ejecuta pytest en tests/test_user.py para verificar'.",
            "good_looks_like": "Un flujo donde cada operación de escritura es seguida por una verificación.",
        },
    },
    "Context": {
        "en": {
            "what_it_is": "Reading the codebase before attempting to write or edit files.",
            "why_it_matters": "Agents don't magically know your whole architecture. They need to read the relevant files first.",
            "how_to_improve": "Ask the agent to grep for usages, or view relevant files before proposing changes.",
            "example": "'Before editing, use grep to find all usages of `UserDTO` in the src/ folder.'",
            "good_looks_like": "The agent reads 3-4 files before editing a central component.",
        },
        "es": {
            "what_it_is": "Leer el código base antes de intentar escribir o editar archivos.",
            "why_it_matters": "Los agentes no conocen tu arquitectura mágicamente. Necesitan leer el contexto.",
            "how_to_improve": "Pide al agente que haga un grep de usos, o que lea archivos clave antes de proponer cambios.",
            "example": "'Antes de editar, haz grep para encontrar todos los usos de `UserDTO` en src/.'",
            "good_looks_like": "El agente lee 3-4 archivos antes de editar un componente central.",
        },
    },
    "Iteration": {
        "en": {
            "what_it_is": "Providing rich feedback when something fails, instead of just 'it failed'.",
            "why_it_matters": "'It failed' gives zero signal. The agent will guess and likely break things further.",
            "how_to_improve": "Paste error traces, explain what you expected to see vs what actually happened.",
            "example": "Instead of 'didn't work', use 'It failed with Error 500 at line 42. I expected it to return an empty array.'",
            "good_looks_like": "Prompting with exact error messages and logic expectations.",
        },
        "es": {
            "what_it_is": "Proveer feedback detallado cuando algo falla, en vez de decir solo 'falló'.",
            "why_it_matters": "'Falló' no da señal. El agente adivinará y probablemente rompa más cosas.",
            "how_to_improve": "Pega trazas de error, explica qué esperabas ver vs qué sucedió.",
            "example": "En lugar de 'no funcionó', usa 'Falló con Error 500 en la línea 42. Esperaba que devolviera un array vacío.'",
            "good_looks_like": "Aportar mensajes de error exactos y expectativas lógicas.",
        },
    },
    "Toolcraft": {
        "en": {
            "what_it_is": "Using the right tools effectively, and delegating autonomy to the agent.",
            "why_it_matters": "Micromanaging (e.g. telling the agent exactly what shell command to run) defeats the purpose of an autonomous agent.",
            "how_to_improve": "State the goal and let the agent figure out the tools. Say 'find where X is defined', not 'run grep -R X .'",
            "example": "'Investigate why the build is failing' instead of 'cat build.log'.",
            "good_looks_like": "The agent leverages multiple tools (grep, view_file, run_command) autonomously to solve a high-level goal.",
        },
        "es": {
            "what_it_is": "Usar las herramientas correctas y delegar autonomía al agente.",
            "why_it_matters": "Hacer micromanagement (decir qué comando exacto correr) anula el propósito de un agente autónomo.",
            "how_to_improve": "Indica la meta y deja que el agente use las herramientas. Di 'encuentra dónde se define X', no 'ejecuta grep -R X .'",
            "example": "'Investiga por qué falla el build' en lugar de 'cat build.log'.",
            "good_looks_like": "El agente usa múltiples herramientas (grep, view_file, run_command) de forma autónoma para resolver un objetivo.",
        },
    },
}
