"""
HYBRID AGENT - Real-Time Learning Literature Review Agent

This is the MAIN agent that combines:
1. Pattern Memory (persistent learning via prompts)
2. Session Memory (in-conversation learning)
3. Background LoRA Training (true weight updates)

WHAT MAKES THIS "HYBRID":
- Instant feedback via memory (you see changes immediately)
- Real learning via LoRA (weights actually update)
- Best of both worlds!

YOU WILL SEE:
- Immediate improvement when you give feedback
- Long-term improvement as model trains
- Persistent learning across sessions
"""

import ollama
from typing import List, Tuple, Optional
import time

from models import Paper
from search import PaperSearcher
from pattern_memory import PatternMemory
from session_memory import SessionMemory, IterativeRefiner
from background_trainer import BackgroundTrainer


class HybridLiteratureAgent:
    """
    Literature review agent with real-time learning
    
    ARCHITECTURE:
    
    ┌────────────────────────────────────────────────┐
    │          HYBRID LEARNING AGENT                 │
    ├────────────────────────────────────────────────┤
    │                                                │
    │  ┌──────────────┐  ┌──────────────┐         │
    │  │   Pattern    │  │   Session    │         │
    │  │   Memory     │  │   Memory     │         │
    │  │ (Persistent) │  │ (Temporary)  │         │
    │  └──────┬───────┘  └──────┬───────┘         │
    │         │                  │                  │
    │         └────────┬─────────┘                 │
    │                  v                            │
    │         ┌────────────────┐                   │
    │         │   LLM Prompt   │                   │
    │         │   (Enhanced)   │                   │
    │         └────────┬───────┘                   │
    │                  v                            │
    │         ┌────────────────┐                   │
    │         │  Generation    │                   │
    │         └────────┬───────┘                   │
    │                  │                            │
    │         ┌────────v───────┐                   │
    │         │  YOUR FEEDBACK │                   │
    │         └────────┬───────┘                   │
    │                  │                            │
    │         ┌────────v────────┐                  │
    │         │ Background LoRA │                  │
    │         │ (Every 5 items) │                  │
    │         └─────────────────┘                  │
    │                                                │
    └────────────────────────────────────────────────┘
    
    LEARNING LAYERS:
    
    Layer 1 (INSTANT): Session Memory
    - Within same conversation
    - "Try again with more stats" → Immediate retry
    - Lasts: Current session only
    
    Layer 2 (PERSISTENT): Pattern Memory
    - Across all sessions
    - "needs stats" → Always include stats
    - Lasts: Forever (saved to disk)
    
    Layer 3 (DEEP): LoRA Training
    - Every 5-10 examples
    - Actual weight updates
    - Lasts: Forever (new model file)
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        lora_trigger: int = 5
    ):
        """
        Initialize hybrid agent
        
        Args:
            model: Ollama model to use
            lora_trigger: Train after this many examples
        
        WHAT HAPPENS:
        1. Initializes all memory systems
        2. Connects to Ollama
        3. Sets up background trainer
        4. Ready to learn!
        """
        
        print("\n" + "="*60)
        print("HYBRID LEARNING AGENT")
        print("="*60)
        
        # Components
        self.model = model
        self.searcher = PaperSearcher()
        
        # Memory systems
        self.pattern_memory = PatternMemory()
        self.session_memory = SessionMemory()
        self.refiner = IterativeRefiner(self.session_memory)
        
        # Background training
        self.background_trainer = BackgroundTrainer(trigger_count=lora_trigger)
        
        # Test Ollama connection
        self._test_ollama()
        
        print("\n✅ Agent ready with hybrid learning!")
        print(f"   Model: {model}")
        print(f"   LoRA trigger: Every {lora_trigger} examples")
        print("="*60 + "\n")
    
    def _test_ollama(self):
        """Test Ollama connection"""
        try:
            ollama.list()
            print(f"✅ Ollama connected")
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            print("   Make sure Ollama is running!")
            raise
    
    def generate_review(
        self,
        query: str,
        papers_per_source: int = 50,
        allow_iteration: bool = True
    ) -> Tuple[str, List[Paper], float, int]:
        """
        Generate literature review with hybrid learning
        
        Args:
            query: Research question
            papers_per_source: Papers from each API
            allow_iteration: Allow in-session iteration
        
        Returns:
            (review, papers, score, rating)
        
        WORKFLOW:
        1. Search for papers
        2. Build enhanced prompt (with learned patterns)
        3. Generate review
        4. Get your feedback
        5. Option to iterate (session memory)
        6. Learn patterns (pattern memory)
        7. Add to training queue (background LoRA)
        
        This is where ALL learning happens!
        """
        
        print(f"\n{'='*60}")
        print(f"GENERATING REVIEW")
        print(f"{'='*60}")
        print(f"Query: {query}\n")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Search for papers
        # ═══════════════════════════════════════════════════════════
        
        papers = self.searcher.search(query, papers_per_source)
        
        if not papers:
            print("❌ No papers found")
            return "No papers found.", [], 0.0, 0
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: Build enhanced prompt with learned patterns
        # ═══════════════════════════════════════════════════════════
        
        # This is KEY - inject learned patterns into prompt!
        # base_prompt = self._build_base_prompt(query, papers)
        # Instead of base_prompt = self._build_base_prompt(query, papers)
        if allow_iteration:  # Or always, for depth
            base_prompt = self._build_enhanced_prompt(query, papers)
        else:
            base_prompt = self._build_base_prompt(query, papers)
        
        # Add pattern memory (persistent learning)
        pattern_instructions = self.pattern_memory.to_prompt_instructions()
        
        # Add session memory (if iterating)
        session_context = self.session_memory.get_context_for_regeneration(query)
        
        # Combined prompt
        # full_prompt = base_prompt + pattern_instructions + session_context
        if 'ENHANCED STRUCTURE' in base_prompt:  # Check if enhanced prompt
            full_prompt = base_prompt + session_context
        else:
            full_prompt = base_prompt + pattern_instructions + session_context
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: Generate review
        # ═══════════════════════════════════════════════════════════
        
        print("✍️  Generating review...")
        if self.pattern_memory.count_patterns() > 0:
            print(f"   ↳ Applying {self.pattern_memory.count_patterns()} learned patterns")
        
        review = self._generate(full_prompt)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: Self-evaluate
        # ═══════════════════════════════════════════════════════════
        
        score = self._self_evaluate(review)
        
        print(f"\n📊 Self-assessment: {score:.1f}/10")
        
        # Record in session memory
        self.session_memory.add_generation(query, review, score)
        
        return review, papers, score, 0  # rating filled by user
    
    def _build_base_prompt(self, query: str, papers: List[Paper]) -> str:
        """
        Build base prompt for review generation
        
        This is the foundation - before adding learned patterns.
        """
        
        # Format papers
        papers_text = ""
        for i, paper in enumerate(papers, 1):
            papers_text += f"\n[{i}] {paper.title}\n"
            papers_text += f"Authors: {', '.join(paper.authors[:5])}\n"
            papers_text += f"Year: {paper.year}\n"
            papers_text += f"Source: {paper.source}\n"
            abstract = paper.abstract[:600] if len(paper.abstract) > 600 else paper.abstract
            papers_text += f"Abstract: {abstract}\n"
            papers_text += "─" * 40 + "\n"
        
        # Base instruction
        prompt = f"""You are an expert academic researcher writing a literature review grounded strictly in the provided papers.

Research Question: {query}

You are given the following retrieved research papers (with metadata and abstracts):
{papers_text}

CRITICAL RULES:

- Use ONLY the papers provided above as evidence.
- Do NOT invent citations, authors, titles, or years.
- Do NOT fabricate references.
- Every claim must be supported by a cited paper.
- If evidence is insufficient, explicitly state uncertainty.
- Do NOT include any disclaimer like "references are fictional".
- Only cite papers using their index numbers like [1], [2], etc.

Writing Instructions:

- Write a cohesive narrative literature review suitable for publication.
- Organize discussion around major research themes or methodological paradigms.
- Synthesize insights across studies rather than summarizing individually.
- Compare approaches, datasets, assumptions, and evaluation strategies.
- Explain how the field has evolved over time.
- Critically analyze strengths, limitations, and unresolved challenges.
- Highlight disagreements or open questions.
- Identify clear research gaps and future directions.
- Exclude tangential papers.

Tone:

- Formal academic prose
- Analytical and precise
- Logical flow

Output Format:

- Write ONLY the review text.
- Do NOT generate a references section.
- Do NOT invent citations.

Begin writing now:"""

        return prompt

    def _build_enhanced_prompt(self, query: str, papers: List[Paper]) -> str:
        """
        Enhanced prompt with explicit structure and learned patterns.
        Builds on _build_base_prompt for deeper, sectioned reviews.
        """
        # Start with the solid base
        base = self._build_base_prompt(query, papers)
        
        # Inject structure and patterns
        enhanced = base + "\n\nENHANCED STRUCTURE (MANDATORY):\n" \
            "Organize your review into these sections:\n" \
            "1. **Overview & Evolution**: Historical context and key milestones [cite 3-5 foundational papers].\n" \
            "2. **Core Methods & Findings**: Synthesize approaches (e.g., compare accuracies: MAE, correlation; include stats like p<0.01) across 10+ papers.\n" \
            "3. **Critical Analysis**: Strengths/limitations, biases (e.g., dataset diversity), contradictions (e.g., 'While [1] reports X, [5] shows Y').\n" \
            "4. **Gaps & Future Directions**: Unresolved issues (e.g., 'No 2024 studies on [topic]') and recommendations.\n" \
            "5. **Conclusion**: Overarching insights and implications.\n\n" \
            "Word count: 1000-1500. End with the review only—no extras."
        
        return enhanced

    def _generate(self, prompt: str) -> str:
        """Call Ollama to generate text"""
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            return f"Error: {e}"
    
    def _self_evaluate(self, review: str) -> float:
        """LLM evaluates own work"""
        
        eval_prompt = f"""Evaluate this literature review on 1-10 scale.

Review:
{review[:1000]}...

Consider:
- Papers cited (15+ is good)
- Methodology analysis present
- Statistical details included
- Critical analysis (not just summary)
- Research gaps identified

Respond with ONLY a number 1-10:"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=eval_prompt
            )
            
            score_text = response['response'].strip()
            score = float(score_text.split()[0])
            return min(max(score, 1.0), 10.0)
        
        except:
            return 5.0
    
    def process_feedback(
        self,
        query: str,
        review: str,
        rating: int,
        feedback: str
    ) -> bool:
        """
        Process user feedback - THIS IS WHERE LEARNING HAPPENS!
        
        Args:
            query: Research query
            review: Generated review
            rating: User's rating 1-10
            feedback: User's feedback text
        
        Returns:
            True if LoRA training triggered
        
        THREE TYPES OF LEARNING:
        
        1. SESSION MEMORY (Instant):
           - Adds feedback to session context
           - Next generation in THIS session uses it
        
        2. PATTERN MEMORY (Persistent):
           - Extracts reusable patterns
           - ALL future sessions use it
        
        3. BACKGROUND LORA (Deep):
           - Adds to training queue
           - Every N examples, triggers real training
        """
        
        print(f"\n{'='*60}")
        print("PROCESSING FEEDBACK (Learning...)")
        print("="*60)
        
        # ───────────────────────────────────────────────────────────
        # LAYER 1: Session Memory (Temporary)
        # ───────────────────────────────────────────────────────────
        
        self.session_memory.add_feedback(feedback, rating)
        print("✓ Session memory updated (immediate effect)")
        
        # ───────────────────────────────────────────────────────────
        # LAYER 2: Pattern Memory (Persistent)
        # ───────────────────────────────────────────────────────────
        
        self.pattern_memory.learn_from_feedback(feedback, rating)
        print("✓ Pattern memory updated (permanent learning)")
        
        # ───────────────────────────────────────────────────────────
        # LAYER 3: Background LoRA (Deep Learning)
        # ───────────────────────────────────────────────────────────
        
        should_train = self.background_trainer.add_example(
            query, review, rating, feedback
        )
        
        if should_train:
            print(f"\n🎓 TRAINING CHECKPOINT REACHED!")
            stats = self.background_trainer.get_stats()
            print(f"   Total examples: {stats['total_examples']}")
            print(f"   Excellent (8+): {stats['excellent_examples']}")
            
            return True  # Caller handles training offer
        
        print("="*60)
        
        return False
    
    def regenerate_with_feedback(
        self,
        query: str,
        papers: List[Paper]
    ) -> Tuple[str, float]:
        """
        Regenerate review using session feedback
        
        This uses SESSION MEMORY for immediate improvement.
        Pattern memory is already injected in base prompt.
        
        Args:
            query: Research query
            papers: Papers to use
        
        Returns:
            (improved_review, score)
        """
        
        print("\n🔄 Regenerating with feedback...")
        
        # Build prompt with all memories
        base_prompt = self._build_base_prompt(query, papers)
        pattern_instructions = self.pattern_memory.to_prompt_instructions()
        session_context = self.session_memory.get_context_for_regeneration(query)
        
        full_prompt = base_prompt + pattern_instructions + session_context
        
        # Generate improved version
        review = self._generate(full_prompt)
        score = self._self_evaluate(review)
        
        # Update session memory
        self.session_memory.add_generation(query, review, score)
        
        print(f"📊 New score: {score:.1f}/10")
        
        return review, score
    
    def show_learned_patterns(self):
        """Display all learned patterns"""
        self.pattern_memory.show_patterns()
    
    def get_training_stats(self) -> dict:
        """Get background training statistics"""
        return self.background_trainer.get_stats()
    
    def prepare_lora_training(self) -> str:
        """
        Prepare for LoRA training
        
        Returns:
            Path to Colab notebook
        """
        return self.background_trainer.prepare_training()


# ═══════════════════════════════════════════════════════════════
# SUMMARY - How Everything Works Together
# ═══════════════════════════════════════════════════════════════

"""
COMPLETE WORKFLOW:

Example 1: "Review CRISPR"
├─ Generate review
├─ You: "Needs more statistics" (Rating: 6)
├─ Session Memory: Adds "needs statistics"
├─ Pattern Memory: Learns "always include statistics"
└─ Background: Saves example #1

Example 2: "Review Quantum Computing"
├─ Pattern Memory injects: "Always include statistics"
├─ Generate review (automatically has statistics!)
├─ You: "Good but needs 2024 papers" (Rating: 7)
├─ Session Memory: Adds "needs 2024 papers"
├─ Pattern Memory: Learns "prioritize recent papers"
└─ Background: Saves example #2

Example 3: "Review AI Ethics"
├─ Pattern Memory injects: "Statistics" + "Recent papers"
├─ Generate review (has both automatically!)
├─ You: "Excellent!" (Rating: 9)
└─ Background: Saves example #3

Example 4: "Review Blockchain"
├─ All patterns apply
├─ You: Rating 8
└─ Background: Saves example #4

Example 5: "Review Climate Change"
├─ All patterns apply
├─ You: Rating 8
└─ Background: "5 examples! Train now?"

You: YES

System: Generates lora_checkpoint_1.ipynb
You: Upload to Colab, run, wait 30 minutes
System: Model trained! Download checkpoint_1.gguf

Example 6: "Review Gene Therapy"
├─ Uses TRAINED model (smarter base!)
├─ PLUS all pattern memory
├─ PLUS session memory
└─ Triple learning! 🚀


LEARNING LAYERS:

┌─────────────────────────────────────────────────────────┐
│ LAYER 1: Session Memory (Instant)                      │
│ - "Try again with stats" → Immediate                   │
│ - Lasts: Current session                                │
│ - Speed: Instant                                         │
│ - Depth: Shallow                                         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ LAYER 2: Pattern Memory (Persistent)                   │
│ - "needs stats" → Always include                       │
│ - Lasts: Forever                                         │
│ - Speed: Instant                                         │
│ - Depth: Medium                                          │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ LAYER 3: LoRA Training (Deep)                          │
│ - Every 5 examples → Real training                     │
│ - Lasts: Forever                                         │
│ - Speed: 30 minutes                                      │
│ - Depth: Deep (weights change!)                         │
└─────────────────────────────────────────────────────────┘

ALL THREE work together!
- Instant feedback (Session)
- Persistent patterns (Pattern Memory)
- True learning (LoRA)

This is HYBRID learning! 🎯
"""
