"""
PATTERN MEMORY - Learns Your Preferences

This module remembers what you want across sessions.
When you say "needs more statistics", it learns to ALWAYS include stats.

KEY CONCEPT:
- Your feedback → Extracted patterns → Applied to future reviews
- Patterns persist across sessions (saved to disk)
- Model "remembers" your preferences without weight updates

Think of it like giving the model a checklist of your preferences
that gets longer as you provide more feedback.
"""

import json
import os
from typing import List, Dict, Set
from datetime import datetime


class PatternMemory:
    """
    Manages learned patterns from user feedback
    
    WHAT THIS DOES:
    1. Extracts reusable patterns from your feedback
       Example: "needs more stats" → "Always include statistical analysis"
    
    2. Saves patterns to disk (pattern_memory.json)
    
    3. Converts patterns to prompt instructions
       Example: "ALWAYS include: statistical analysis, recent papers, ..."
    
    4. Injects into every generation (model follows the rules)
    
    WHAT THIS DOES NOT DO:
    - Change model weights (that's LoRA's job)
    - Require training time
    - Need GPU
    
    This is INSTANT learning via prompting!
    """
    
    def __init__(self, filepath: str = "memory/pattern_memory.json"):
        """
        Initialize pattern memory
        
        Args:
            filepath: Where to save/load patterns
        
        WHAT HAPPENS:
        1. Sets file path
        2. Initializes pattern categories
        3. Tries to load existing patterns from disk
        4. If no file exists, starts empty
        """
        self.filepath = filepath
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Pattern categories
        # These are the "rules" the model will follow
        self.patterns = {
            # Things to ALWAYS include
            "always_include": set(),
            # Example: {"statistical analysis", "recent papers", "methodology critique"}
            
            # Things to AVOID
            "avoid": set(),
            # Example: {"vague statements", "unsupported claims"}
            
            # Quality thresholds
            "quality_thresholds": {},
            # Example: {"min_papers": 15, "min_citations": 20}
            
            # Style preferences
            "style_preferences": set(),
            # Example: {"critical analysis", "compare and contrast"}
            
            # Specific requirements
            "requirements": set(),
            # Example: {"cite sources as [1][2]", "include executive summary"}
        }
        
        # Load existing patterns (if any)
        self.load()
        
        print(f"📚 Pattern Memory initialized")
        print(f"   Loaded {self.count_patterns()} patterns")
    
    def learn_from_feedback(self, feedback: str, rating: int):
        """
        Extract patterns from user feedback
        
        This is the KEY function - turns your feedback into reusable rules!
        
        Args:
            feedback: What you said (e.g., "needs more statistics")
            rating: Your rating 1-10
        
        HOW IT WORKS:
        1. Analyzes feedback text for keywords
        2. Extracts patterns based on keywords
        3. Adds to appropriate category
        4. Saves to disk
        
        EXAMPLES:
        
        Feedback: "needs more statistics"
        → Pattern: always_include.add("statistical results with p-values")
        
        Feedback: "too vague"
        → Pattern: avoid.add("vague or general statements")
        
        Feedback: "only 8 papers cited"
        → Pattern: quality_thresholds["min_papers"] = 15
        
        Feedback: "needs critical analysis"
        → Pattern: style_preferences.add("critical analysis comparing findings")
        """
        
        feedback_lower = feedback.lower()
        
        # ═══════════════════════════════════════════════════════════
        # PATTERN EXTRACTION - Keyword matching
        # ═══════════════════════════════════════════════════════════
        
        # ─────────────────────────────────────────────────────────
        # 1. STATISTICS-related patterns
        # ─────────────────────────────────────────────────────────
        if any(word in feedback_lower for word in ['statistic', 'p-value', 'p<', 'significance', 'n=', 'effect size']):
            self.patterns["always_include"].add(
                "Include specific statistical results (p-values, effect sizes, sample sizes)"
            )
            print("   ✓ Learned: Always include statistical details")
        
        # ─────────────────────────────────────────────────────────
        # 2. RECENT PAPERS patterns
        # ─────────────────────────────────────────────────────────
        if any(word in feedback_lower for word in ['recent', '2024', '2023', 'latest', 'current']):
            self.patterns["always_include"].add(
                "Prioritize recent papers from 2023-2024"
            )
            print("   ✓ Learned: Focus on recent papers")
        
        # ─────────────────────────────────────────────────────────
        # 3. METHODOLOGY patterns
        # ─────────────────────────────────────────────────────────
        if any(word in feedback_lower for word in ['methodology', 'method', 'study design', 'experimental']):
            self.patterns["always_include"].add(
                "Analyze research methodology and study design"
            )
            print("   ✓ Learned: Include methodology analysis")
        
        # ─────────────────────────────────────────────────────────
        # 4. CRITICAL ANALYSIS patterns
        # ─────────────────────────────────────────────────────────
        if any(word in feedback_lower for word in ['critical', 'critique', 'analyze', 'compare', 'contrast']):
            self.patterns["style_preferences"].add(
                "Provide critical analysis comparing and contrasting findings"
            )
            print("   ✓ Learned: Use critical analysis style")
        
        # ─────────────────────────────────────────────────────────
        # 5. PAPER COUNT patterns
        # ─────────────────────────────────────────────────────────
        if any(word in feedback_lower for word in ['more papers', 'few papers', 'only']):
            # Try to extract number
            # Example: "only 8 papers" → set min to 15
            if 'few' in feedback_lower or 'more' in feedback_lower:
                self.patterns["quality_thresholds"]["min_papers"] = 15
                print("   ✓ Learned: Minimum 15 papers required")
        
        # ─────────────────────────────────────────────────────────
        # 6. VAGUENESS patterns (what to AVOID)
        # ─────────────────────────────────────────────────────────
        if any(word in feedback_lower for word in ['vague', 'general', 'unclear', 'specific']):
            self.patterns["avoid"].add(
                "Vague or general statements without specific evidence"
            )
            print("   ✓ Learned: Avoid vague statements")
        
        # ─────────────────────────────────────────────────────────
        # 7. RESEARCH GAPS patterns
        # ─────────────────────────────────────────────────────────
        if any(word in feedback_lower for word in ['gap', 'future', 'limitation', 'missing']):
            self.patterns["always_include"].add(
                "Identify research gaps and future directions"
            )
            print("   ✓ Learned: Always identify research gaps")
        
        # ─────────────────────────────────────────────────────────
        # 8. CITATION FORMAT patterns
        # ─────────────────────────────────────────────────────────
        if any(word in feedback_lower for word in ['citation', 'cite', 'reference', '[1]', '[2]']):
            self.patterns["requirements"].add(
                "Cite all claims using [1], [2], [3] format"
            )
            print("   ✓ Learned: Proper citation format required")
        
        # ─────────────────────────────────────────────────────────
        # 9. COMPREHENSIVE patterns
        # ─────────────────────────────────────────────────────────
        if any(word in feedback_lower for word in ['comprehensive', 'thorough', 'detailed', 'depth']):
            self.patterns["style_preferences"].add(
                "Provide comprehensive and detailed analysis"
            )
            print("   ✓ Learned: Comprehensive style preferred")
        
        # ─────────────────────────────────────────────────────────
        # 10. Rating-based thresholds
        # ─────────────────────────────────────────────────────────
        # If rating is low, it means something is systematically wrong
        if rating <= 5:
            # Add general quality requirement
            self.patterns["requirements"].add(
                "Ensure high-quality analysis meeting all standards"
            )
        
        # Save updated patterns to disk
        self.save()
    
    def to_prompt_instructions(self) -> str:
        """
        Convert patterns to prompt that LLM will follow
        
        Returns:
            String of instructions to inject into prompts
        
        HOW THIS WORKS:
        Takes all learned patterns and formats them as instructions.
        These get added to EVERY prompt, so the model follows your rules!
        
        Example output:
        '''
        Follow these learned user preferences:
        
        ALWAYS INCLUDE:
        - Statistical results with p-values
        - Recent papers from 2023-2024
        - Methodology analysis
        
        AVOID:
        - Vague statements
        
        QUALITY REQUIREMENTS:
        - Minimum 15 papers
        
        STYLE:
        - Critical analysis
        - Comprehensive detail
        '''
        """
        
        # If no patterns learned yet, return empty
        if self.count_patterns() == 0:
            return ""
        
        # Build instruction string
        instructions = "\n" + "="*60 + "\n"
        instructions += "LEARNED USER PREFERENCES (follow these strictly):\n"
        instructions += "="*60 + "\n"
        
        # ─────────────────────────────────────────────────────────
        # 1. Things to ALWAYS include
        # ─────────────────────────────────────────────────────────
        if self.patterns["always_include"]:
            instructions += "\n✓ ALWAYS INCLUDE:\n"
            for item in sorted(self.patterns["always_include"]):
                instructions += f"  - {item}\n"
        
        # ─────────────────────────────────────────────────────────
        # 2. Things to AVOID
        # ─────────────────────────────────────────────────────────
        if self.patterns["avoid"]:
            instructions += "\n✗ AVOID:\n"
            for item in sorted(self.patterns["avoid"]):
                instructions += f"  - {item}\n"
        
        # ─────────────────────────────────────────────────────────
        # 3. Quality thresholds
        # ─────────────────────────────────────────────────────────
        if self.patterns["quality_thresholds"]:
            instructions += "\n📊 QUALITY REQUIREMENTS:\n"
            for key, value in self.patterns["quality_thresholds"].items():
                instructions += f"  - {key}: {value}\n"
        
        # ─────────────────────────────────────────────────────────
        # 4. Style preferences
        # ─────────────────────────────────────────────────────────
        if self.patterns["style_preferences"]:
            instructions += "\n✍️  STYLE:\n"
            for item in sorted(self.patterns["style_preferences"]):
                instructions += f"  - {item}\n"
        
        # ─────────────────────────────────────────────────────────
        # 5. Specific requirements
        # ─────────────────────────────────────────────────────────
        if self.patterns["requirements"]:
            instructions += "\n📋 REQUIREMENTS:\n"
            for item in sorted(self.patterns["requirements"]):
                instructions += f"  - {item}\n"
        
        instructions += "="*60 + "\n"
        
        return instructions
    
    def save(self):
        """
        Save patterns to disk
        
        WHAT HAPPENS:
        1. Converts sets to lists (JSON doesn't support sets)
        2. Writes to pattern_memory.json
        3. Patterns persist across sessions!
        """
        # Convert sets to lists for JSON serialization
        serializable = {
            "always_include": list(self.patterns["always_include"]),
            "avoid": list(self.patterns["avoid"]),
            "quality_thresholds": self.patterns["quality_thresholds"],
            "style_preferences": list(self.patterns["style_preferences"]),
            "requirements": list(self.patterns["requirements"]),
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def load(self):
        """
        Load patterns from disk
        
        WHAT HAPPENS:
        1. Try to read pattern_memory.json
        2. If exists: convert lists back to sets, restore patterns
        3. If not exists: start with empty patterns
        
        This runs when you start the system - loads your learned preferences!
        """
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to sets
            self.patterns["always_include"] = set(data.get("always_include", []))
            self.patterns["avoid"] = set(data.get("avoid", []))
            self.patterns["quality_thresholds"] = data.get("quality_thresholds", {})
            self.patterns["style_preferences"] = set(data.get("style_preferences", []))
            self.patterns["requirements"] = set(data.get("requirements", []))
        
        except FileNotFoundError:
            # No saved patterns yet - that's fine, start empty
            pass
    
    def count_patterns(self) -> int:
        """Count total number of learned patterns"""
        return (
            len(self.patterns["always_include"]) +
            len(self.patterns["avoid"]) +
            len(self.patterns["quality_thresholds"]) +
            len(self.patterns["style_preferences"]) +
            len(self.patterns["requirements"])
        )
    
    def show_patterns(self):
        """Display all learned patterns (for user to see)"""
        print("\n" + "="*60)
        print("LEARNED PATTERNS")
        print("="*60)
        
        if self.count_patterns() == 0:
            print("\nNo patterns learned yet.")
            print("Provide feedback to start teaching the model!\n")
            return
        
        print(self.to_prompt_instructions())
    
    def reset(self):
        """Clear all learned patterns (start fresh)"""
        for key in self.patterns:
            if isinstance(self.patterns[key], set):
                self.patterns[key] = set()
            elif isinstance(self.patterns[key], dict):
                self.patterns[key] = {}
        
        self.save()
        print("✓ All patterns cleared")


# ═══════════════════════════════════════════════════════════════
# SUMMARY - How Pattern Memory Works
# ═══════════════════════════════════════════════════════════════

"""
WORKFLOW:

1. You give feedback:
   "needs more statistics"

2. Pattern extracted:
   always_include.add("Include statistical results")

3. Saved to disk:
   pattern_memory.json updated

4. Next generation:
   Prompt includes: "ALWAYS INCLUDE: Statistical results"

5. Model follows the rule:
   Generated review includes statistics!

6. Pattern persists:
   Even after restart, model remembers!


KEY BENEFITS:

✅ INSTANT: No training time, patterns apply immediately
✅ PERSISTENT: Survives session restarts
✅ CUMULATIVE: Patterns accumulate as you give more feedback
✅ TRANSPARENT: You can see all learned patterns
✅ REVERSIBLE: Can reset if needed


LIMITATIONS:

❌ Not "true" learning (just prompt engineering)
❌ Limited by context window
❌ Can't learn complex patterns
❌ Doesn't update model weights

For TRUE learning, we need LoRA (next component!)
"""
