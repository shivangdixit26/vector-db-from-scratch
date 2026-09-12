"""
text_corpus.py — Load real text corpus and compute embeddings.

Supports two modes:
1. Load 5,000 AG News headlines from data/corpus.txt (preferred)
2. Fall back to ~200 hardcoded headlines if corpus.txt is missing

Embeddings are computed with sentence-transformers (MiniLM-L6-v2)
and cached to data/cache/embeddings.npy for fast reload.
"""

import os
import numpy as np


# ── Hardcoded fallback corpus (200 diverse headlines) ────────────────────
FALLBACK_HEADLINES = [
    # Finance & Economy
    "Global stock markets rally on positive economic data",
    "Federal Reserve signals potential rate cut amid slowing inflation",
    "Cryptocurrency market experiences unprecedented volatility",
    "Tech stocks surge following strong quarterly earnings reports",
    "Oil prices plummet as OPEC members disagree on production cuts",
    "Housing market shows signs of cooling in major metropolitan areas",
    "Central banks worldwide coordinate emergency interest rate decisions",
    "Trade deficit narrows significantly in the last quarter",
    "Unemployment rate drops to historic low of three percent",
    "Small business optimism index reaches five year high",
    "Electric vehicle sales surpass traditional auto sales for first time",
    "Government announces new stimulus package for renewable energy",
    "Major bank merger creates largest financial institution in history",
    "Inflation expectations fall as consumer confidence rises",
    "Agricultural commodity prices spike due to extreme weather events",
    "International trade agreements reshape global supply chains",
    "Retail sales exceed expectations during holiday shopping season",
    "Venture capital investment in AI startups reaches record levels",
    "Currency markets react to unexpected geopolitical developments",
    "National debt approaches critical threshold economists warn",
    # Technology
    "Revolutionary AI system achieves human level reasoning capabilities",
    "Quantum computing breakthrough solves previously impossible problems",
    "New smartphone features advanced health monitoring sensors",
    "Social media platform faces scrutiny over content moderation policies",
    "Autonomous vehicle completes first cross country journey",
    "Scientists develop brain computer interface for paralysis patients",
    "Cybersecurity attack exposes millions of user accounts",
    "Space tourism company announces first commercial lunar mission",
    "Renewable energy technology achieves grid parity in most markets",
    "Virtual reality headset sales double compared to previous year",
    "New programming language designed specifically for AI development",
    "Researchers create self healing electronic materials",
    "Cloud computing market reaches one trillion dollar valuation",
    "Blockchain technology transforms international payment systems",
    "Open source software foundation receives record corporate funding",
    "Wearable technology detects health emergencies before symptoms appear",
    "Internet of Things devices surpass fifty billion worldwide",
    "Machine learning model predicts natural disasters with high accuracy",
    "Digital twins technology revolutionizes manufacturing processes",
    "Augmented reality glasses enter mainstream consumer market",
    # Science & Health
    "New treatment shows remarkable results against treatment resistant depression",
    "Mars rover discovers evidence of ancient microbial life",
    "Gene therapy successfully treats inherited blindness in clinical trial",
    "Scientists achieve nuclear fusion energy milestone",
    "Antibiotic resistant bacteria identified in hospital environments",
    "Climate scientists report accelerating ice sheet melting rates",
    "Vaccine developed for previously untreatable tropical disease",
    "Astronomers detect mysterious radio signals from distant galaxy",
    "Stem cell therapy regenerates damaged heart tissue in patients",
    "Deep sea exploration reveals entirely new ecosystem",
    "CRISPR gene editing technique advances to human clinical trials",
    "New study links gut microbiome to mental health outcomes",
    "Coral reef restoration project shows promising early results",
    "Particle physics experiment discovers unexpected subatomic behavior",
    "Air pollution levels reach dangerous thresholds in major cities",
    "Plastic eating bacteria could transform waste management",
    "Brain imaging study reveals new understanding of consciousness",
    "Renewable energy powers entire nation for record continuous period",
    "Cancer immunotherapy achieves complete remission in late stage patients",
    "Oceanographic survey maps previously unexplored deep sea trenches",
    # Sports
    "Underdog team wins championship in dramatic overtime victory",
    "Olympic athlete breaks long standing world record in swimming",
    "Major league announces expansion with two new franchise teams",
    "Star player signs record breaking contract extension",
    "International soccer tournament draws billions of viewers worldwide",
    "Tennis champion announces unexpected retirement from professional sport",
    "College basketball tournament delivers historic upsets in first round",
    "Formula One team unveils revolutionary aerodynamic car design",
    "Professional esports league attracts mainstream television coverage",
    "Marathon runner completes race despite challenging weather conditions",
    "Cricket world cup final ends in a thrilling super over",
    "Golf major championship won by first time major winner",
    "Women professional basketball league announces historic media deal",
    "Mixed martial arts fighter wins bout with spectacular knockout",
    "Winter sports federation introduces new extreme event category",
    "Youth development program produces multiple professional athletes",
    "Stadium construction project sets new sustainability standards",
    "Athletes protest for social justice causes during ceremonies",
    "Fantasy sports industry reaches new participation milestones",
    "Track and field association updates anti doping testing protocols",
    # Entertainment & Culture
    "Blockbuster movie shatters opening weekend box office records",
    "Streaming service subscribers surpass one billion globally",
    "Award winning director announces ambitious new film trilogy",
    "Music festival attendance reaches all time high this summer",
    "Bestselling novel adapted into critically acclaimed television series",
    "Video game launch generates more revenue than any movie release",
    "Art exhibition explores intersection of technology and creativity",
    "Podcast industry revenue doubles for the third consecutive year",
    "Celebrity chef opens innovative restaurant concept in new city",
    "Broadway musical wins record number of theater awards",
    "Independent film wins top prize at international film festival",
    "Music streaming reshapes how artists earn and distribute royalties",
    "Cultural heritage site receives UNESCO world heritage designation",
    "Documentary film sparks national conversation about social issues",
    "Animation studio pioneers new visual effects technology",
    "Book publishing industry adapts to changing digital landscape",
    "Comedy special becomes most watched content on streaming platform",
    "Museum attendance increases following free admission policy",
    "Fashion industry commits to sustainable production practices",
    "Digital art marketplace creates new revenue streams for artists",
    # Politics & World Affairs
    "International summit addresses urgent climate change legislation",
    "Election results signal significant shift in political landscape",
    "Peace negotiations resume between long time rival nations",
    "Immigration policy reform passes through legislative committee",
    "Diplomatic relations restored after decades of tensions",
    "Healthcare reform bill advances to final vote in senate",
    "International humanitarian aid reaches crisis affected regions",
    "Education funding bill receives bipartisan congressional support",
    "Cybersecurity alliance formed between allied nations",
    "Constitutional amendment proposed for voting rights expansion",
    "Trade embargo lifted following diplomatic breakthrough",
    "Emergency session called to address humanitarian crisis",
    "New environmental regulations target industrial emissions",
    "Defense spending bill passes with significant modifications",
    "International court issues landmark ruling on digital privacy",
    "Economic sanctions imposed following human rights violations",
    "Coalition government formed after inconclusive election results",
    "Refugee resettlement program expands to additional countries",
    "Anti corruption commission launches sweeping investigation",
    "Infrastructure development plan receives international funding",
    # Education
    "University develops free online courses reaching millions worldwide",
    "Study reveals significant achievement gap in standardized testing",
    "Teachers union negotiates improved working conditions agreement",
    "New educational technology transforms special needs instruction",
    "Community college enrollment surges amid economic uncertainty",
    "Research university receives largest donation in its history",
    "Student loan reform legislation advances through congress",
    "International student enrollment reaches record levels",
    "Early childhood education program shows lasting positive effects",
    "Vocational training programs address critical workforce shortages",
    # Environment
    "Massive reforestation project plants one billion trees globally",
    "Ocean cleanup initiative removes record amount of plastic waste",
    "Electric public transportation expands across urban centers",
    "Endangered species population rebounds following conservation efforts",
    "Carbon capture technology deployed at industrial scale",
    "Sustainable agriculture practices gain widespread adoption",
    "Water conservation program reduces urban consumption significantly",
    "Biodiversity survey discovers hundreds of previously unknown species",
    "Green building certification becomes standard for new construction",
    "Environmental justice movement achieves landmark policy victories",
    # Innovation & Startups
    "Drone delivery service launches in multiple major cities",
    "Three D printing technology creates functional human organs",
    "Vertical farming startup raises massive funding round",
    "Personalized medicine approach shows improved patient outcomes",
    "Smart city infrastructure reduces energy consumption dramatically",
    "Autonomous shipping vessels begin commercial ocean crossings",
    "Lab grown meat receives regulatory approval for consumer sales",
    "Neural network architecture achieves breakthrough in language understanding",
    "Robotic surgery system performs increasingly complex procedures",
    "Satellite internet service provides connectivity to remote areas",
    # Weather & Natural Events
    "Unprecedented heat wave breaks temperature records across continent",
    "Hurricane season forecast predicts above average tropical activity",
    "Earthquake early warning system successfully alerts millions",
    "Drought conditions threaten agricultural output in major regions",
    "Northern lights visible at unusually low latitudes",
    "Volcanic eruption disrupts air travel across multiple countries",
    "Flooding devastates communities following extreme rainfall events",
    "Wildfire containment efforts make significant progress",
    "Severe winter storm causes widespread power outages",
    "Tornado outbreak affects communities across multiple states",
]


class TextCorpus:
    """
    Loads a text corpus and computes/caches sentence embeddings.
    
    Priority:
    1. Load from data/corpus.txt (5,000 AG News headlines)
    2. Fall back to hardcoded 200 headlines
    
    Embeddings are cached to data/cache/embeddings.npy.
    """

    def __init__(self, corpus_path: str = "data/corpus.txt",
                 cache_dir: str = "data/cache",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir
        self.model_name = model_name
        self._model = None
        
        # Load texts
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                self.texts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(self.texts)} texts from {corpus_path}")
        else:
            self.texts = list(FALLBACK_HEADLINES)
            print(f"Using fallback corpus ({len(self.texts)} headlines)")
            print(f"  Run 'python prepare_corpus.py' for the full 5,000 text corpus.")

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _cache_path(self) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        n = len(self.texts)
        return os.path.join(self.cache_dir, f"embeddings_{n}.npy")

    def load_or_create(self) -> np.ndarray:
        """
        Load cached embeddings or compute them from scratch.
        Returns (n, dim) float32 array.
        """
        cache_path = self._cache_path()
        
        if os.path.exists(cache_path):
            vectors = np.load(cache_path)
            if vectors.shape[0] == len(self.texts):
                print(f"Loaded cached embeddings from {cache_path}")
                return vectors
            print(f"Cache size mismatch ({vectors.shape[0]} vs {len(self.texts)}), recomputing...")
        
        print(f"Computing embeddings for {len(self.texts)} texts...")
        vectors = self.model.encode(
            self.texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True
        )
        vectors = vectors.astype(np.float32)
        
        np.save(cache_path, vectors)
        print(f"Embeddings cached to {cache_path}")
        print(f"Shape: {vectors.shape}, dtype: {vectors.dtype}")
        
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns (dim,) float32 vector."""
        vec = self.model.encode([query], normalize_embeddings=True)
        return vec[0].astype(np.float32)
