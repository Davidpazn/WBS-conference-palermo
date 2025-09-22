"""
Diagram generation tool for E2B sandboxes
-----------------------------------------
Generates diagrams (Mermaid, Graphviz) inside E2B sandboxes and retrieves them
as artifacts. Supports SVG and PNG formats with proper validation and security.

Features:
- Mermaid and Graphviz diagram generation
- Input validation and sanitization
- Artifact storage and retrieval
- Telemetry integration
- Security safeguards against code injection
"""

import os
import re
import json
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from opentelemetry import trace

@dataclass
class DiagramSpec:
    """Validated diagram specification"""
    engine: Literal["mermaid", "dot"]
    title: str
    spec: str
    format: Literal["svg", "png"]

    def validate(self) -> Dict[str, Any]:
        """Validate diagram specification for security and correctness"""
        issues = []

        # Length limits
        if len(self.spec) > 10000:
            issues.append("Diagram specification too long (max 10000 chars)")

        if len(self.title) > 200:
            issues.append("Title too long (max 200 chars)")

        # Security checks - look for dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'onclick\s*=',
            r'onerror\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'subprocess',
            r'__import__',
            r'file://',
            r'http[s]?://',
            r'\\x[0-9a-fA-F]{2}',  # hex encoding
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, self.spec, re.IGNORECASE):
                issues.append(f"Potentially dangerous pattern detected: {pattern}")

        # Engine-specific validation
        if self.engine == "mermaid":
            # Basic Mermaid syntax validation
            if not any(keyword in self.spec.lower() for keyword in [
                'graph', 'flowchart', 'sequencediagram', 'classdiagram',
                'statediagram', 'pie', 'journey', 'gantt'
            ]):
                issues.append("Mermaid diagram doesn't contain recognized diagram type")

        elif self.engine == "dot":
            # Basic Graphviz DOT validation
            if not re.search(r'(graph|digraph)\s+\w*\s*\{', self.spec):
                issues.append("DOT diagram doesn't contain valid graph/digraph declaration")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

class DiagramGenerator:
    """Diagram generation tool with E2B integration"""

    def __init__(self):
        self.tracer = trace.get_tracer(__name__)

    def propose_diagram_json(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Use LLM to propose a diagram specification based on query and context
        Returns strict JSON format as required by NB3 spec
        """
        with self.tracer.start_as_current_span("diagram.propose") as span:
            span.set_attribute("diagram.query", query[:200])
            span.set_attribute("diagram.context_length", len(context))

            try:
                from openai import OpenAI

                client = OpenAI()
                model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

                system_prompt = """You are a diagram design expert. Given a query and context, propose a diagram specification.

OUTPUT REQUIREMENTS:
- Return ONLY a valid JSON object with these exact fields:
  - "engine": either "mermaid" or "dot"
  - "title": short descriptive title (max 100 chars)
  - "spec": the diagram code/specification
  - "format": either "svg" or "png"

DIAGRAM GUIDELINES:
- For flowcharts, processes, sequences: use "mermaid"
- For network graphs, hierarchies, dependencies: use "dot" (Graphviz)
- Keep diagrams simple and readable
- Use meaningful node labels
- Prefer "svg" format for scalability

SECURITY:
- NO script tags, javascript, or executable code
- NO external URLs or file references
- NO shell commands or system calls
- Only pure diagram specification code

Example mermaid:
{"engine":"mermaid","title":"User Registration Flow","spec":"flowchart TD\\n    A[Start] --> B[Enter Email]\\n    B --> C{Valid Email?}\\n    C -->|Yes| D[Send Verification]\\n    C -->|No| B","format":"svg"}

Example dot:
{"engine":"dot","title":"System Architecture","spec":"digraph G {\\n    rankdir=TB;\\n    A [label=\\"Frontend\\"];\\n    B [label=\\"API\\"];\\n    C [label=\\"Database\\"];\\n    A -> B;\\n    B -> C;\\n}","format":"svg"}"""

                user_prompt = f"Query: {query}\n\nContext: {context[:1000] if context else 'No additional context'}"

                with self.tracer.start_as_current_span("gen_ai.diagram_proposal") as gen_span:
                    gen_span.set_attribute("gen_ai.system", "openai")
                    gen_span.set_attribute("gen_ai.operation.name", "responses")
                    gen_span.set_attribute("gen_ai.request.model", model)

                    # 1) Use function calling to get structured output
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0,
                        max_tokens=800,
                        tools=[{
                            "type": "function",
                            "function": {
                                "name": "propose_diagram",
                                "description": "Propose a diagram specification",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "engine": {"type": "string", "enum": ["mermaid", "dot"]},
                                        "title":  {"type": "string", "maxLength": 100},
                                        "spec":   {"type": "string", "minLength": 1},
                                        "format": {"type": "string", "enum": ["svg", "png"]}
                                    },
                                    "required": ["engine", "title", "spec", "format"],
                                    "additionalProperties": False
                                }
                            }
                        }],
                        tool_choice={"type": "function", "function": {"name": "propose_diagram"}}
                    )

                    # 2) Extract the function call arguments
                    tool_call = response.choices[0].message.tool_calls[0]
                    response_text = tool_call.function.arguments

                    # 3) Parse the JSON directly (already clean from function calling)
                    s = response_text.strip()
                    response_text = s

                    gen_span.set_attribute("gen_ai.completion", response_text[:500])

                # Parse JSON response
                try:
                    diagram_json = json.loads(response_text)

                    # Validate required fields
                    required_fields = ["engine", "title", "spec", "format"]
                    for field in required_fields:
                        if field not in diagram_json:
                            raise ValueError(f"Missing required field: {field}")

                    # Validate field values
                    if diagram_json["engine"] not in ["mermaid", "dot"]:
                        raise ValueError("Invalid engine, must be 'mermaid' or 'dot'")

                    if diagram_json["format"] not in ["svg", "png"]:
                        raise ValueError("Invalid format, must be 'svg' or 'png'")

                    span.set_attribute("diagram.engine", diagram_json["engine"])
                    span.set_attribute("diagram.format", diagram_json["format"])
                    span.set_attribute("diagram.spec_length", len(diagram_json["spec"]))

                    return {
                        "success": True,
                        "diagram": diagram_json
                    }

                except json.JSONDecodeError as e:
                    span.set_attribute("error", True)
                    return {
                        "success": False,
                        "error": f"Invalid JSON response: {e}",
                        "raw_response": response_text
                    }
                except ValueError as e:
                    span.set_attribute("error", True)
                    return {
                        "success": False,
                        "error": str(e),
                        "raw_response": response_text
                    }

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", True)
                return {
                    "success": False,
                    "error": f"Failed to generate diagram proposal: {e}"
                }

    def render_diagram_e2b(self, diagram_spec: DiagramSpec, sandbox=None) -> Dict[str, Any]:
        """
        Render diagram in E2B sandbox and return artifact information

        Args:
            diagram_spec: Validated diagram specification
            sandbox: Active E2B sandbox instance (if None, creates new one)

        Returns:
            Dict with success status, remote_path, bytes, and metadata
        """
        with self.tracer.start_as_current_span("diagram.render") as span:
            span.set_attribute("diagram.engine", diagram_spec.engine)
            span.set_attribute("diagram.format", diagram_spec.format)
            span.set_attribute("diagram.spec_length", len(diagram_spec.spec))

            # Validate diagram first
            validation = diagram_spec.validate()
            if not validation["valid"]:
                span.set_attribute("error", True)
                return {
                    "success": False,
                    "error": "Diagram validation failed",
                    "validation_issues": validation["issues"]
                }

            try:
                from e2b_code_interpreter import Sandbox

                # Use provided sandbox or create new one
                if sandbox is None:
                    create_new_sandbox = True
                    sbx = Sandbox.create()
                else:
                    create_new_sandbox = False
                    sbx = sandbox

                try:
                    span.set_attribute("sandbox.id", sbx.sandbox_id)
                    span.set_attribute("sandbox.created_new", create_new_sandbox)

                    # Create diagrams directory
                    sbx.run_code("import os; os.makedirs('/home/user/diagrams', exist_ok=True)")

                    if diagram_spec.engine == "mermaid":
                        # Install Mermaid CLI using npx (avoids global install)
                        # First ensure npm is initialized
                        sbx.run("bash -lc 'npm init -y >/dev/null 2>&1 || true'")

                        # Write Mermaid specification
                        with open('/tmp/diagram_spec.mmd', 'w') as f:
                            f.write(diagram_spec.spec)

                        # Upload to sandbox
                        sbx.upload_file('/tmp/diagram_spec.mmd', '/home/user/diagrams/input.mmd')

                        # Generate diagram using npx
                        output_path = f"/home/user/diagrams/diagram.{diagram_spec.format}"
                        render_cmd = f"npx -y @mermaid-js/mermaid-cli -i /home/user/diagrams/input.mmd -o {output_path}"
                        result = sbx.run(f"bash -lc '{render_cmd}'")

                    elif diagram_spec.engine == "dot":
                        # Install Graphviz if needed
                        install_cmd = "apt-get update && apt-get install -y graphviz 2>/dev/null || true"
                        sbx.run(f"bash -lc '{install_cmd}'")

                        # Write DOT specification
                        with open('/tmp/diagram_spec.dot', 'w') as f:
                            f.write(diagram_spec.spec)

                        # Upload to sandbox
                        sbx.upload_file('/tmp/diagram_spec.dot', '/home/user/diagrams/input.dot')

                        # Generate diagram
                        output_path = f"/home/user/diagrams/diagram.{diagram_spec.format}"
                        render_cmd = f"dot -T{diagram_spec.format} /home/user/diagrams/input.dot -o {output_path}"
                        result = sbx.run(f"bash -lc '{render_cmd}'")

                    # Check if file was created
                    check_result = sbx.run(f"bash -lc 'ls -la {output_path}'")

                    if "No such file" in str(check_result.logs.stderr):
                        return {
                            "success": False,
                            "error": "Diagram file was not created",
                            "render_output": str(result.logs.stderr) if result.logs else str(result)
                        }

                    # Download the generated diagram
                    diagram_bytes = sbx.download_file(output_path)

                    span.set_attribute("diagram.output_size_bytes", len(diagram_bytes))
                    span.set_attribute("diagram.remote_path", output_path)

                    # Clean up temp files
                    try:
                        os.remove('/tmp/diagram_spec.mmd')
                    except:
                        pass
                    try:
                        os.remove('/tmp/diagram_spec.dot')
                    except:
                        pass

                    return {
                        "success": True,
                        "remote_path": output_path,
                        "bytes": diagram_bytes,
                        "size_bytes": len(diagram_bytes),
                        "title": diagram_spec.title,
                        "engine": diagram_spec.engine,
                        "format": diagram_spec.format
                    }

                finally:
                    # Only close sandbox if we created it
                    if create_new_sandbox:
                        sbx.close()

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", True)
                return {
                    "success": False,
                    "error": f"Diagram rendering failed: {e}"
                }

# Convenience functions
def create_diagram_spec(engine: str, title: str, spec: str, format: str = "svg") -> DiagramSpec:
    """Create and validate a diagram specification"""
    return DiagramSpec(
        engine=engine,
        title=title,
        spec=spec,
        format=format
    )

def generate_diagram_e2b(query: str, context: str = "", sandbox=None) -> Dict[str, Any]:
    """End-to-end diagram generation: propose + render"""
    generator = DiagramGenerator()

    # Propose diagram
    proposal = generator.propose_diagram_json(query, context)
    if not proposal["success"]:
        return proposal

    # Create specification
    diagram_data = proposal["diagram"]
    spec = create_diagram_spec(
        engine=diagram_data["engine"],
        title=diagram_data["title"],
        spec=diagram_data["spec"],
        format=diagram_data["format"]
    )

    # Validate
    validation = spec.validate()
    if not validation["valid"]:
        return {
            "success": False,
            "error": "Generated diagram failed validation",
            "validation_issues": validation["issues"]
        }

    # Render
    result = generator.render_diagram_e2b(spec, sandbox)
    result["proposal"] = proposal["diagram"]

    return result