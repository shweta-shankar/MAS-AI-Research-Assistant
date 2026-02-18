"""
Simple data models for literature review
"""

from dataclasses import dataclass
from typing import List, Optional
import json
from datetime import datetime


@dataclass
class Paper:
    """A research paper"""
    title: str
    authors: List[str]
    year: str
    abstract: str
    source: str  # "PubMed", "arXiv", "CrossRef"
    
    # Optional fields
    doi: Optional[str] = None
    url: Optional[str] = None
    journal: Optional[str] = None
    
    def short_citation(self) -> str:
        """Short citation for display"""
        authors_str = ", ".join(self.authors[:2])
        if len(self.authors) > 2:
            authors_str += " et al."
        return f"{authors_str} ({self.year}). {self.title}"


@dataclass  
class TrainingExample:
    """One training example from human feedback"""
    query: str
    review: str
    your_rating: int  # 1-10
    your_feedback: str
    papers_count: int
    timestamp: str
    
    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'query': self.query,
            'review': self.review,
            'your_rating': self.your_rating,
            'your_feedback': self.your_feedback,
            'papers_count': self.papers_count,
            'timestamp': self.timestamp
        }
    
    @staticmethod
    def from_dict(data):
        """Load from dictionary"""
        return TrainingExample(
            query=data['query'],
            review=data['review'],
            your_rating=data['your_rating'],
            your_feedback=data['your_feedback'],
            papers_count=data['papers_count'],
            timestamp=data['timestamp']
        )


class TrainingData:
    """Manages collection of training examples"""
    
    def __init__(self, filepath: str = "training_data.json"):
        self.filepath = filepath
        self.examples: List[TrainingExample] = []
        self.load()
    
    def add(self, example: TrainingExample):
        """Add a training example"""
        self.examples.append(example)
        self.save()
    
    def save(self):
        """Save to file"""
        data = [ex.to_dict() for ex in self.examples]
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f" Saved to {self.filepath}")
    
    def load(self):
        """Load from file"""
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                self.examples = [TrainingExample.from_dict(d) for d in data]
        except FileNotFoundError:
            self.examples = []
    
    def get_excellent(self, min_rating: int = 8) -> List[TrainingExample]:
        """Get high-rated examples"""
        return [ex for ex in self.examples if ex.your_rating >= min_rating]
    
    def stats(self) -> dict:
        """Get statistics"""
        if not self.examples:
            return {"total": 0}
        
        ratings = [ex.your_rating for ex in self.examples]
        excellent = self.get_excellent()
        
        return {
            "total": len(self.examples),
            "excellent_8plus": len(excellent),
            "average_rating": sum(ratings) / len(ratings),
            "min_rating": min(ratings),
            "max_rating": max(ratings)
        }
