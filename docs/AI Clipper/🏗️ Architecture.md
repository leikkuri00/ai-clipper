# Architecture

> High-level design of the AI Clipper Master Prompt engine.

## System overview

```mermaid
flowchart TB
    subgraph Input
        URL[Video URL]
        Local[Local File]
    end

    subgraph Pipeline
        DL[Downloader]
        Audio[Audio Extractor]
        STT[Whisper Transcriber]
        CT[Content Type Classifier]
        SS[Semantic Segmenter]
        MMA[Multimodal Analyzer]
        Scorer[26-Dim Scorer]
        CG[Candidate Generator]
        RP[Retention Predictor]
        CO[Clip Optimizer]
        SSB[Story Series Builder]
        Mod[Moderation]
        CC[Clip Cutter]
        Gal[HTML Gallery]
    end

    subgraph Output
        SC[8 Standalone Clips]
        SER[3 Story Series]
        REP[Report JSON]
        IDX[index.html]
    end

    URL --> DL
    Local --> DL
    DL --> Audio
    Audio --> STT
    STT --> CT
    STT --> SS
    SS --> MMA
    SS --> Scorer
    MMA --> Scorer
    Scorer --> CG
    CG --> RP
    RP --> CO
    CO --> SC
    SS --> SSB
    SSB --> SER
    SC --> Mod
    SER --> Mod
    Mod --> CC
    CC --> Gal
    SSB --> REP
    CO --> REP
    Gal --> IDX
```

## Module responsibilities

| Module | File | Responsibility |
|---|---|---|
| `AIClipper` | `pipeline.py` | Orchestrates the entire pipeline |
| `ClipperConfig` | `config.py` | All tunable parameters |
| `SemanticSegmenter` | `semantic_segmenter.py` | Splits transcript by meaning |
| `ViralScorer` | `scorer.py` | 26-dimension LLM scoring |
| `CandidateGenerator` | `candidate_generator.py` | 300+ clip candidates |
| `RetentionPredictor` | `retention_predictor.py` | Engagement predictions |
| `ClipOptimizer` | `clip_optimizer.py` | Iterative boundary refinement |
| `StorySeriesBuilder` | `story_series_builder.py` | 3 continuous series |
| `MultimodalAnalyzer` | `multimodal_analyzer.py` | Audio + video signals |
| `transcriber` | `transcriber.py` | Whisper transcription |
| `clip_cutter` | `clip_cutter.py` | FFmpeg clip rendering |
| `html_gallery` | `html_gallery.py` | Output gallery |

## Data flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as clip_tool.py / web_app.py
    participant Pipe as AIClipper
    participant LLM as LM Studio
    participant FS as Filesystem

    User->>CLI: URL or file
    CLI->>Pipe: run()
    Pipe->>FS: download / load video
    Pipe->>FS: extract audio
    Pipe->>LLM: classify content type
    Pipe->>LLM: semantic segmentation
    Pipe->>Pipe: multimodal analysis
    Pipe->>LLM: score each segment (26 dims)
    Pipe->>Pipe: generate candidates
    Pipe->>Pipe: heuristic candidate scoring
    Pipe->>LLM: optimize top candidates
    Pipe->>LLM: build story series
    Pipe->>LLM: moderate selected clips
    Pipe->>FS: render clips with FFmpeg
    Pipe->>FS: write report + gallery
    CLI->>User: output folder
```

## Scoring weights

```mermaid
pie title Master Prompt Overall Score Weights
    "Retention" : 30
    "Story" : 20
    "Hook" : 15
    "Emotion" : 10
    "Education" : 10
    "Visual" : 5
    "Audio" : 5
    "Virality" : 5
```

## Key design decisions

1. **Local-first**: Defaults to LM Studio/Ollama; cloud is optional.
2. **Semantic over temporal**: Segmentation is meaning-driven, not time-driven.
3. **Fast candidate scoring**: Heuristic overlap scoring avoids 300+ LLM calls.
4. **Fallbacks everywhere**: Sentence boundaries, continuous arcs, and heuristic scores keep the pipeline running if LLM calls fail.
5. **Diversity enforcement**: Standalone clip selection spreads across topics and emotions.

---

#architecture #ai-clipper #system-design
