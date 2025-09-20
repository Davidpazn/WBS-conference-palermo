"""
Compliance Gate Implementation for AI Agent Flows
Maps to NB3-compliance.txt specification and infra/compliance.yaml config
"""

import re
import yaml
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ComplianceAction(str, Enum):
    """Actions the compliance gate can take"""
    BLOCK = "BLOCK"
    WARN = "WARN"
    FIX = "FIX"


class ComplianceViolation(BaseModel):
    """Represents a compliance rule violation"""
    rule_id: str
    category: str
    action: ComplianceAction
    description: str
    details: str
    violating_content: Optional[str] = None


class ComplianceResult(BaseModel):
    """Result of compliance check"""
    passed: bool
    violations: List[ComplianceViolation] = Field(default_factory=list)
    fixed_content: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class ComplianceGate:
    """Main compliance gate implementation"""
    
    def __init__(self, config_path: str = "src/infra/compliance.yaml"):
        self.config = self._load_config(config_path)
        self.enabled = self.config.get("global_settings", {}).get("enabled", True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load compliance configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Compliance config not found at {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback configuration if YAML not found"""
        return {
            "global_settings": {"enabled": True, "log_violations": True},
            "model_controls": {},
            "tool_safety": {},
            "memory_privacy": {},
            "output_quality": {},
            "trading_constraints": {}
        }
    
    def check_state(self, state: Dict[str, Any]) -> ComplianceResult:
        """Main compliance check for agent state"""
        if not self.enabled:
            return ComplianceResult(passed=True)
        
        result = ComplianceResult(passed=True)
        
        # A) Model & Cost Controls
        self._check_model_controls(state, result)
        
        # B) Tool & Network Safety  
        self._check_tool_safety(state, result)
        
        # C) Memory & Privacy
        self._check_memory_privacy(state, result)
        
        # D) Output Quality
        self._check_output_quality(state, result)
        
        # E) Trading Constraints
        self._check_trading_constraints(state, result)
        
        # Set overall pass/fail
        blocking_violations = [v for v in result.violations if v.action == ComplianceAction.BLOCK]
        result.passed = len(blocking_violations) == 0
        
        if self.config.get("global_settings", {}).get("log_violations", True):
            self._log_violations(result.violations)
        
        return result
    
    def _check_model_controls(self, state: Dict[str, Any], result: ComplianceResult):
        """Check model and cost controls (CMP-MOD-*)"""
        meta = state.get("meta", {})
        controls = self.config.get("model_controls", {})
        
        # CMP-MOD-001: Model allowlist
        if "CMP-MOD-001" in controls:
            model = meta.get("model", "")
            allowed = controls["CMP-MOD-001"].get("allowed_models", [])
            if model and model not in allowed:
                result.violations.append(ComplianceViolation(
                    rule_id="CMP-MOD-001",
                    category="model_controls",
                    action=ComplianceAction.BLOCK,
                    description="Model not in allowlist",
                    details=f"Model '{model}' not in {allowed}",
                    violating_content=model
                ))
        
        # CMP-MOD-002: Token limits
        if "CMP-MOD-002" in controls:
            config = controls["CMP-MOD-002"]
            input_tokens = meta.get("input_tokens", 0)
            output_tokens = meta.get("output_tokens", 0)
            
            if input_tokens > config.get("max_input_tokens", 50000):
                result.violations.append(ComplianceViolation(
                    rule_id="CMP-MOD-002",
                    category="model_controls", 
                    action=ComplianceAction.BLOCK,
                    description="Input tokens exceed limit",
                    details=f"Input tokens: {input_tokens} > {config['max_input_tokens']}"
                ))
            
            if output_tokens > config.get("max_output_tokens", 16000):
                result.violations.append(ComplianceViolation(
                    rule_id="CMP-MOD-002",
                    category="model_controls",
                    action=ComplianceAction.BLOCK, 
                    description="Output tokens exceed limit",
                    details=f"Output tokens: {output_tokens} > {config['max_output_tokens']}"
                ))
        
        # CMP-MOD-003: Cost budget
        if "CMP-MOD-003" in controls:
            cost = state.get("cost", 0.0)
            max_cost = controls["CMP-MOD-003"].get("max_cost_eur", 2.50)
            if cost > max_cost:
                result.violations.append(ComplianceViolation(
                    rule_id="CMP-MOD-003",
                    category="model_controls",
                    action=ComplianceAction.BLOCK,
                    description="Cost exceeds budget",
                    details=f"Cost: €{cost} > €{max_cost}"
                ))
        
        # CMP-MOD-004: Temperature bounds
        if "CMP-MOD-004" in controls:
            config = controls["CMP-MOD-004"]
            temperature = meta.get("temperature")
            node_name = meta.get("current_node", "")
            applies_to = config.get("applies_to_nodes", [])
            
            if temperature is not None and node_name in applies_to:
                min_temp = config.get("min_temperature", 0.0)
                max_temp = config.get("max_temperature", 0.7)
                if not (min_temp <= temperature <= max_temp):
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-MOD-004",
                        category="model_controls",
                        action=ComplianceAction.WARN,
                        description="Temperature out of bounds",
                        details=f"Temperature {temperature} not in [{min_temp}, {max_temp}] for node {node_name}"
                    ))

    def _check_tool_safety(self, state: Dict[str, Any], result: ComplianceResult):
        """Check tool and network safety (CMP-TOOL-*)"""
        meta = state.get("meta", {})
        safety = self.config.get("tool_safety", {})
        
        # CMP-TOOL-001: Domain allowlist
        if "CMP-TOOL-001" in safety:
            web_calls = meta.get("web_calls", [])
            allowed_domains = safety["CMP-TOOL-001"].get("allowed_domains", [])
            
            for call in web_calls:
                domain = self._extract_domain(call.get("url", ""))
                if domain and domain not in allowed_domains:
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-TOOL-001",
                        category="tool_safety",
                        action=ComplianceAction.BLOCK,
                        description="Domain not in allowlist",
                        details=f"Domain '{domain}' not in {allowed_domains}",
                        violating_content=call.get("url")
                    ))
        
        # CMP-TOOL-002: Shell command denylist
        if "CMP-TOOL-002" in safety:
            shell_commands = meta.get("shell_commands", [])
            denied_substrings = safety["CMP-TOOL-002"].get("denied_substrings", [])
            
            for cmd in shell_commands:
                for denied in denied_substrings:
                    if denied in cmd:
                        result.violations.append(ComplianceViolation(
                            rule_id="CMP-TOOL-002",
                            category="tool_safety",
                            action=ComplianceAction.BLOCK,
                            description="Dangerous shell command",
                            details=f"Command contains '{denied}'",
                            violating_content=cmd
                        ))
        
        # CMP-TOOL-003: File write scope
        if "CMP-TOOL-003" in safety:
            file_writes = meta.get("file_writes", [])
            allowed_paths = safety["CMP-TOOL-003"].get("allowed_paths", [])
            denied_paths = safety["CMP-TOOL-003"].get("denied_paths", [])
            
            for file_path in file_writes:
                path_allowed = any(file_path.startswith(allowed) for allowed in allowed_paths)
                path_denied = any(file_path.startswith(denied) for denied in denied_paths)
                
                if path_denied or not path_allowed:
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-TOOL-003",
                        category="tool_safety",
                        action=ComplianceAction.BLOCK,
                        description="File write outside allowed scope",
                        details=f"Path '{file_path}' not in allowed paths or in denied paths",
                        violating_content=file_path
                    ))
        
        # CMP-TOOL-004: E2B isolation
        if "CMP-TOOL-004" in safety:
            code_executed = meta.get("code_executed", False)
            sandbox = meta.get("sandbox", "")
            required_sandbox = safety["CMP-TOOL-004"].get("required_sandbox", "e2b")
            
            if code_executed and sandbox != required_sandbox:
                result.violations.append(ComplianceViolation(
                    rule_id="CMP-TOOL-004",
                    category="tool_safety",
                    action=ComplianceAction.WARN,
                    description="Code execution not in required sandbox",
                    details=f"Expected sandbox '{required_sandbox}', got '{sandbox}'"
                ))

    def _check_memory_privacy(self, state: Dict[str, Any], result: ComplianceResult):
        """Check memory and privacy (CMP-MEM-*)"""
        memory_data = state.get("memory_data", {})
        privacy = self.config.get("memory_privacy", {})
        
        # CMP-MEM-001: Field allowlist
        if "CMP-MEM-001" in privacy and memory_data:
            allowed_fields = privacy["CMP-MEM-001"].get("allowed_fields", [])
            for field in memory_data.keys():
                if field not in allowed_fields:
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-MEM-001",
                        category="memory_privacy",
                        action=ComplianceAction.WARN,
                        description="Memory field not in allowlist",
                        details=f"Field '{field}' not in {allowed_fields}"
                    ))
        
        # CMP-MEM-002: PII patterns
        if "CMP-MEM-002" in privacy and memory_data:
            pii_regexes = privacy["CMP-MEM-002"].get("pii_regexes", [])
            content = str(memory_data)
            
            for pattern in pii_regexes:
                if re.search(pattern, content, re.IGNORECASE):
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-MEM-002",
                        category="memory_privacy",
                        action=ComplianceAction.BLOCK,
                        description="PII detected in memory content",
                        details=f"Pattern '{pattern}' matched in memory data"
                    ))
        
        # CMP-MEM-003: Secret leakage
        if "CMP-MEM-003" in privacy and memory_data:
            secret_patterns = privacy["CMP-MEM-003"].get("secret_patterns", [])
            content = str(memory_data)
            
            for pattern in secret_patterns:
                if re.search(pattern, content):
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-MEM-003",
                        category="memory_privacy",
                        action=ComplianceAction.BLOCK,
                        description="Secret detected in memory content",
                        details=f"Secret pattern matched in memory data",
                        violating_content="[REDACTED]"
                    ))
        
        # CMP-MEM-004: Size and TTL
        if "CMP-MEM-004" in privacy and memory_data:
            config = privacy["CMP-MEM-004"]
            content_size = len(str(memory_data).encode('utf-8'))
            max_bytes = config.get("max_item_bytes", 8192)
            
            if content_size > max_bytes:
                result.violations.append(ComplianceViolation(
                    rule_id="CMP-MEM-004",
                    category="memory_privacy",
                    action=ComplianceAction.WARN,
                    description="Memory item exceeds size limit",
                    details=f"Size: {content_size} bytes > {max_bytes} bytes"
                ))

    def _check_output_quality(self, state: Dict[str, Any], result: ComplianceResult):
        """Check output and claims quality (CMP-OUT-*)"""
        output_text = state.get("result", "")
        if not output_text:
            return
            
        quality = self.config.get("output_quality", {})
        
        # CMP-OUT-001: Required disclaimer
        if "CMP-OUT-001" in quality:
            config = quality["CMP-OUT-001"]
            trigger_keywords = config.get("trigger_keywords", [])
            disclaimer = config.get("disclaimer", "")
            
            needs_disclaimer = any(keyword.lower() in output_text.lower() for keyword in trigger_keywords)
            has_disclaimer = disclaimer in output_text
            
            if needs_disclaimer and not has_disclaimer:
                # Auto-fix: prepend disclaimer
                fixed_text = f"{disclaimer}\n\n{output_text}"
                result.fixed_content = fixed_text
                result.warnings.append(f"Auto-inserted disclaimer for financial content")
        
        # CMP-OUT-002: Banned phrases
        if "CMP-OUT-002" in quality:
            banned_phrases = quality["CMP-OUT-002"].get("banned_phrases", [])
            
            for phrase in banned_phrases:
                if phrase.lower() in output_text.lower():
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-OUT-002",
                        category="output_quality",
                        action=ComplianceAction.BLOCK,
                        description="Banned phrase detected",
                        details=f"Output contains banned phrase: '{phrase}'",
                        violating_content=phrase
                    ))
        
        # CMP-OUT-003: Citations on facts
        if "CMP-OUT-003" in quality:
            meta = state.get("meta", {})
            requires_citations = meta.get("requires_citations", False)
            
            if requires_citations:
                citation_patterns = quality["CMP-OUT-003"].get("citation_patterns", [])
                has_citations = any(re.search(pattern, output_text) for pattern in citation_patterns)
                
                if not has_citations:
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-OUT-003",
                        category="output_quality",
                        action=ComplianceAction.WARN,
                        description="External facts require citations",
                        details="Output makes factual claims but has no citations"
                    ))
        
        # CMP-OUT-004: Toxicity screening
        if "CMP-OUT-004" in quality:
            toxic_patterns = quality["CMP-OUT-004"].get("toxic_patterns", [])
            
            for pattern in toxic_patterns:
                if re.search(pattern, output_text, re.IGNORECASE):
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-OUT-004",
                        category="output_quality",
                        action=ComplianceAction.WARN,
                        description="Potentially toxic content detected",
                        details=f"Pattern '{pattern}' matched in output"
                    ))

    def _check_trading_constraints(self, state: Dict[str, Any], result: ComplianceResult):
        """Check trading-specific constraints (CMP-TRD-*)"""
        trades = state.get("trades", [])
        if not trades:
            return
            
        constraints = self.config.get("trading_constraints", {})
        
        # CMP-TRD-001: Max single asset weight
        if "CMP-TRD-001" in constraints:
            max_weight = constraints["CMP-TRD-001"].get("max_single_asset_weight", 0.25)
            
            for trade in trades:
                weight = trade.get("weight", 0.0)
                if weight > max_weight:
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-TRD-001",
                        category="trading_constraints",
                        action=ComplianceAction.BLOCK,
                        description="Single asset weight exceeds limit",
                        details=f"Asset {trade.get('symbol')} weight {weight} > {max_weight}",
                        violating_content=str(trade)
                    ))
        
        # CMP-TRD-002: Minimum cash
        if "CMP-TRD-002" in constraints:
            min_cash = constraints["CMP-TRD-002"].get("min_cash_weight", 0.05)
            cash_weight = sum(t.get("weight", 0) for t in trades if t.get("symbol") == "CASH")
            
            if cash_weight < min_cash:
                result.violations.append(ComplianceViolation(
                    rule_id="CMP-TRD-002",
                    category="trading_constraints",
                    action=ComplianceAction.BLOCK,
                    description="Cash position below minimum",
                    details=f"Cash weight {cash_weight} < {min_cash}"
                ))
        
        # CMP-TRD-003: Prohibited assets
        if "CMP-TRD-003" in constraints:
            prohibited = constraints["CMP-TRD-003"].get("prohibited_assets", [])
            
            for trade in trades:
                symbol = trade.get("symbol", "")
                if any(prohibited_asset in symbol for prohibited_asset in prohibited):
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-TRD-003",
                        category="trading_constraints",
                        action=ComplianceAction.BLOCK,
                        description="Trade in prohibited asset",
                        details=f"Symbol '{symbol}' contains prohibited asset type",
                        violating_content=str(trade)
                    ))
        
        # CMP-TRD-004: Leverage and shorting
        if "CMP-TRD-004" in constraints:
            config = constraints["CMP-TRD-004"]
            max_total = config.get("max_total_weight", 1.0)
            allow_shorting = config.get("allow_shorting", False)
            
            total_weight = sum(t.get("weight", 0) for t in trades)
            if total_weight > max_total:
                result.violations.append(ComplianceViolation(
                    rule_id="CMP-TRD-004",
                    category="trading_constraints",
                    action=ComplianceAction.BLOCK,
                    description="Total weight exceeds maximum (leverage)",
                    details=f"Total weight {total_weight} > {max_total}"
                ))
            
            if not allow_shorting:
                short_trades = [t for t in trades if t.get("quantity", 0) < 0]
                if short_trades:
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-TRD-004",
                        category="trading_constraints",
                        action=ComplianceAction.BLOCK,
                        description="Short selling not allowed",
                        details=f"Found {len(short_trades)} short positions"
                    ))
        
        # CMP-TRD-005: Turnover cap
        if "CMP-TRD-005" in constraints:
            max_turnover = constraints["CMP-TRD-005"].get("max_turnover", 0.50)
            total_turnover = sum(abs(t.get("weight_change", 0)) for t in trades)
            
            if total_turnover > max_turnover:
                result.violations.append(ComplianceViolation(
                    rule_id="CMP-TRD-005",
                    category="trading_constraints",
                    action=ComplianceAction.BLOCK,
                    description="Portfolio turnover exceeds cap",
                    details=f"Turnover {total_turnover} > {max_turnover}"
                ))
        
        # CMP-TRD-006: Lot size and precision
        if "CMP-TRD-006" in constraints:
            config = constraints["CMP-TRD-006"]
            min_lot = config.get("min_lot_size", 1)
            max_decimals = config.get("max_decimals", 6)
            
            for trade in trades:
                quantity = trade.get("quantity", 0)
                if abs(quantity) < min_lot:
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-TRD-006",
                        category="trading_constraints",
                        action=ComplianceAction.BLOCK,
                        description="Trade quantity below minimum lot size",
                        details=f"Quantity {quantity} < {min_lot}",
                        violating_content=str(trade)
                    ))
                
                # Check decimal precision
                decimal_places = len(str(quantity).split('.')[-1]) if '.' in str(quantity) else 0
                if decimal_places > max_decimals:
                    result.violations.append(ComplianceViolation(
                        rule_id="CMP-TRD-006",
                        category="trading_constraints",
                        action=ComplianceAction.BLOCK,
                        description="Trade quantity has too many decimal places",
                        details=f"Quantity {quantity} has {decimal_places} decimals > {max_decimals}",
                        violating_content=str(trade)
                    ))

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        import urllib.parse
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc
        except:
            return ""
    
    def _log_violations(self, violations: List[ComplianceViolation]):
        """Log compliance violations"""
        for violation in violations:
            level = logging.ERROR if violation.action == ComplianceAction.BLOCK else logging.WARNING
            logger.log(level, f"Compliance violation {violation.rule_id}: {violation.description} - {violation.details}")


def create_compliance_gate(config_path: str = "src/infra/compliance.yaml") -> ComplianceGate:
    """Factory function to create compliance gate"""
    return ComplianceGate(config_path)


# Utility functions for nodes to record metadata
def record_model_usage(state: Dict[str, Any], model: str, input_tokens: int, output_tokens: int, cost: float, temperature: float = None):
    """Record model usage in state metadata for compliance checking"""
    if "meta" not in state:
        state["meta"] = {}
    
    state["meta"].update({
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "temperature": temperature
    })
    
    if "cost" not in state:
        state["cost"] = 0.0
    state["cost"] += cost


def record_web_call(state: Dict[str, Any], url: str, method: str = "GET"):
    """Record web call for compliance checking"""
    if "meta" not in state:
        state["meta"] = {}
    
    if "web_calls" not in state["meta"]:
        state["meta"]["web_calls"] = []
    
    state["meta"]["web_calls"].append({
        "url": url,
        "method": method
    })


def record_shell_command(state: Dict[str, Any], command: str):
    """Record shell command for compliance checking"""
    if "meta" not in state:
        state["meta"] = {}
    
    if "shell_commands" not in state["meta"]:
        state["meta"]["shell_commands"] = []
    
    state["meta"]["shell_commands"].append(command)


def record_file_write(state: Dict[str, Any], file_path: str):
    """Record file write for compliance checking"""
    if "meta" not in state:
        state["meta"] = {}
    
    if "file_writes" not in state["meta"]:
        state["meta"]["file_writes"] = []
    
    state["meta"]["file_writes"].append(file_path)


def record_code_execution(state: Dict[str, Any], sandbox: str = "local"):
    """Record code execution for compliance checking"""
    if "meta" not in state:
        state["meta"] = {}
    
    state["meta"]["code_executed"] = True
    state["meta"]["sandbox"] = sandbox


def record_memory_data(state: Dict[str, Any], data: Dict[str, Any]):
    """Record memory data for compliance checking"""
    state["memory_data"] = data


def record_trades(state: Dict[str, Any], trades: List[Dict[str, Any]]):
    """Record trades for compliance checking"""
    state["trades"] = trades