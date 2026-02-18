"""
SESSION MEMORY - Temporary In-Conversation Learning

This module handles INSTANT feedback within a session.
When you say "try again with more stats", it regenerates immediately.

KEY CONCEPT:
- Feedback → Added to conversation context → Immediate regeneration
- Only lasts for current session (not persistent like PatternMemory)
- Used for rapid iteration on same query

Think of it like a conversation where the model remembers what
you said 2 minutes ago, but not what you said yesterday.
"""

from typing import List, Dict, Optional
from datetime import datetime


class SessionMemory:
    """
    Manages feedback within a single training session
    
    WHAT THIS DOES:
    1. Keeps track of all feedback in current session
    2. Allows immediate iteration on same query
    3. Builds conversation-style context for LLM
    
    EXAMPLE WORKFLOW:
    
    You: "Review CRISPR"
    Agent: [generates review A]
    You: "Bad! Needs more stats. Try again."
    
    SessionMemory adds your feedback to context
    
    Agent: [generates review B with "add more stats" in mind]
    You: "Better! But needs 2024 papers."
    
    SessionMemory adds this too
    
    Agent: [generates review C with both feedbacks]
    
    LIFESPAN:
    - Starts when you run training script
    - Ends when you close the script
    - Does NOT persist to next session
    
    For persistence, use PatternMemory!
    """
    
    def __init__(self):
        """
        Initialize session memory
        
        WHAT HAPPENS:
        - Creates empty conversation history
        - Ready to track feedback
        """
        # Conversation history for this session
        # Format: List of messages like a chat
        self.conversation = []
        
        # Feedback items (just text)
        self.feedback_items = []
        
        # Start time
        self.session_start = datetime.now()
        
        print("💭 Session Memory active (in-conversation learning enabled)")
    
    def add_generation(self, query: str, review: str, score: float):
        """
        Record a generation
        
        Args:
            query: The research query
            review: Generated review
            score: Self-assessed score
        
        WHAT THIS DOES:
        Adds the generation to conversation history.
        This allows model to "remember" what it generated before.
        """
        self.conversation.append({
            "type": "generation",
            "query": query,
            "review": review,
            "score": score,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_feedback(self, feedback: str, rating: int):
        """
        Add user feedback to session
        
        Args:
            feedback: What you said
            rating: Your rating 1-10
        
        WHAT THIS DOES:
        1. Adds feedback to list
        2. Marks as "needs attention"
        3. Will be used in next generation
        
        EXAMPLE:
        feedback = "needs more statistical analysis"
        → Next prompt includes: "User wants: more statistical analysis"
        """
        self.feedback_items.append({
            "feedback": feedback,
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        })
        
        self.conversation.append({
            "type": "feedback",
            "content": feedback,
            "rating": rating
        })
    
    def get_context_for_regeneration(self, current_query: str) -> str:
        """
        Build context string for regenerating with feedback
        
        Args:
            current_query: The query being regenerated
        
        Returns:
            Context string to add to prompt
        
        HOW THIS WORKS:
        Takes all feedback from this session and formats it
        as instructions for the LLM.
        
        EXAMPLE OUTPUT:
        '''
        IMPORTANT - User provided this feedback on previous attempt:
        - "Needs more statistical analysis" (rated 5/10)
        - "Include 2024 papers" (rated 6/10)
        
        Please address ALL this feedback in your revised review.
        '''
        """
        
        if not self.feedback_items:
            return ""
        
        context = "\n" + "="*60 + "\n"
        context += "⚠️  REGENERATION - Address previous feedback:\n"
        context += "="*60 + "\n"
        
        for item in self.feedback_items:
            context += f"\n• \"{item['feedback']}\" (user rated previous: {item['rating']}/10)\n"
        
        context += "\nPlease address ALL this feedback in your revised review.\n"
        context += "="*60 + "\n"
        
        return context
    
    def get_recent_feedback(self, n: int = 3) -> List[str]:
        """
        Get most recent feedback items
        
        Args:
            n: How many recent items to get
        
        Returns:
            List of recent feedback strings
        
        USED FOR:
        Quick summary of what user wants.
        Can be shown in UI or added to prompt.
        """
        recent = self.feedback_items[-n:] if len(self.feedback_items) >= n else self.feedback_items
        return [item["feedback"] for item in recent]
    
    def should_regenerate(self, last_rating: int, threshold: int = 7) -> bool:
        """
        Decide if should offer regeneration
        
        Args:
            last_rating: User's rating of current review
            threshold: Minimum acceptable rating
        
        Returns:
            True if should offer to try again
        
        LOGIC:
        If user rates below threshold, they're unhappy.
        Offer to regenerate with their feedback.
        """
        return last_rating < threshold
    
    def clear(self):
        """
        Clear session memory
        
        USE CASES:
        - Start fresh on new topic
        - Reset after many iterations
        - Clean slate for testing
        """
        self.conversation = []
        self.feedback_items = []
        print("💭 Session memory cleared")
    
    def summary(self) -> Dict:
        """
        Get session summary
        
        Returns:
            Dictionary with session stats
        
        SHOWS:
        - How many generations
        - How many feedback items
        - Average rating
        - Session duration
        """
        generations = [c for c in self.conversation if c["type"] == "generation"]
        feedbacks = [c for c in self.conversation if c["type"] == "feedback"]
        
        ratings = [f["rating"] for f in feedbacks] if feedbacks else []
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        duration = (datetime.now() - self.session_start).seconds
        
        return {
            "generations": len(generations),
            "feedback_items": len(feedbacks),
            "average_rating": avg_rating,
            "duration_seconds": duration
        }


class IterativeRefiner:
    """
    Handles iterative refinement with feedback
    
    WHAT THIS DOES:
    Manages the "try again" loop:
    1. Generate
    2. Get feedback
    3. Regenerate with feedback
    4. Repeat until satisfied
    
    Uses SessionMemory to track conversation flow.
    """
    
    def __init__(self, session_memory: SessionMemory):
        """
        Initialize refiner
        
        Args:
            session_memory: SessionMemory instance to use
        """
        self.session_memory = session_memory
    
    def should_offer_retry(self, rating: int, iteration: int, max_iterations: int = 3) -> bool:
        """
        Decide if should offer to try again
        
        Args:
            rating: User's rating
            iteration: Current iteration number
            max_iterations: Max allowed iterations
        
        Returns:
            True if should offer retry
        
        LOGIC:
        - If rating low (< 7) AND not at max iterations → offer retry
        - If rating high (>= 7) → satisfied, no retry
        - If max iterations reached → stop
        """
        return rating < 7 and iteration < max_iterations
    
    def build_refinement_prompt(self, original_prompt: str, review: str, feedback: str) -> str:
        """
        Build prompt for refinement
        
        Args:
            original_prompt: Original generation prompt
            review: Previous review generated
            feedback: User's feedback on it
        
        Returns:
            New prompt that includes feedback
        
        HOW THIS WORKS:
        Takes original prompt and adds:
        - What was generated before
        - What user said was wrong
        - Instruction to fix it
        
        EXAMPLE:
        original_prompt: "Review CRISPR gene editing..."
        feedback: "needs more statistics"
        
        Returns:
        '''
        Review CRISPR gene editing...
        
        PREVIOUS ATTEMPT:
        [previous review text]
        
        USER FEEDBACK:
        "needs more statistics"
        
        Please generate an IMPROVED review addressing this feedback.
        '''
        """
        
        refinement = original_prompt
        refinement += "\n\n" + "="*60 + "\n"
        refinement += "REFINEMENT REQUESTED\n"
        refinement += "="*60 + "\n"
        refinement += f"\nPrevious attempt had issues:\n{feedback}\n"
        refinement += "\nPlease generate an IMPROVED review addressing this feedback.\n"
        refinement += "="*60 + "\n"
        
        return refinement


# ═══════════════════════════════════════════════════════════════
# SUMMARY - How Session Memory Works
# ═══════════════════════════════════════════════════════════════

"""
SESSION MEMORY vs PATTERN MEMORY:

┌──────────────────────────────────────────────────────────────┐
│ SESSION MEMORY (Temporary)                                   │
├──────────────────────────────────────────────────────────────┤
│ Duration: Current session only                               │
│ Purpose: Rapid iteration on same query                       │
│ Example: "Try again with more stats" → Immediate retry      │
│ Persistence: NO - cleared on exit                            │
│ Use case: Refining a single review                           │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ PATTERN MEMORY (Persistent)                                  │
├──────────────────────────────────────────────────────────────┤
│ Duration: Forever (saved to disk)                            │
│ Purpose: Learn general preferences                           │
│ Example: "needs stats" → Always include stats in future     │
│ Persistence: YES - survives restarts                         │
│ Use case: Teaching model your standards                      │
└──────────────────────────────────────────────────────────────┘


WORKFLOW EXAMPLE:

Query: "Review CRISPR"

Iteration 1:
  Generate → Show review
  You: "Bad! Only 8 papers. Needs 15+. Rating: 4/10"
  SessionMemory.add_feedback("Needs 15+ papers", 4)
  Offer: "Want to try again with this feedback? (y/n)"
  You: "y"

Iteration 2:
  context = SessionMemory.get_context_for_regeneration()
  # context includes: "User wants 15+ papers"
  Generate with context → Show improved review
  You: "Better! But needs more stats. Rating: 6/10"
  SessionMemory.add_feedback("Needs more statistics", 6)
  Offer: "Want to try again? (y/n)"
  You: "y"

Iteration 3:
  context = SessionMemory.get_context_for_regeneration()
  # context includes: "15+ papers" AND "more statistics"
  Generate with context → Show improved review
  You: "Good! Rating: 8/10"
  
  PatternMemory.learn_from_feedback("needs statistics", 8)
  # Now ALL future reviews will include statistics!

Next query: "Review Quantum Computing"
  PatternMemory injects: "Always include statistics"
  Review automatically has statistics from the start!


KEY INSIGHT:
- SessionMemory: Temporary, for refining THIS review
- PatternMemory: Permanent, for ALL future reviews
- Together: Instant feedback + Long-term learning
"""
