import matplotlib.pyplot as plt
import numpy as np
import json

def set_publication_style():
    """Apply a sleek, modern, dark-mode presentation theme."""
    plt.rcParams.update({
        'figure.facecolor': '#0f172a',      # Slate 900
        'axes.facecolor': '#1e293b',        # Slate 800
        'savefig.facecolor': '#0f172a',
        'text.color': '#f8fafc',             # Slate 50
        'axes.labelcolor': '#cbd5e1',        # Slate 300
        'xtick.color': '#94a3b8',           # Slate 400
        'ytick.color': '#94a3b8',
        'grid.color': '#334155',            # Slate 700
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.edgecolor': '#475569',
        'axes.linewidth': 1.2,
    })

def plot_tradeoff(results: list[dict], save_path: str = 'results/accuracy_vs_speed.png') -> None:
    """
    Plot high-impact speed-accuracy tradeoff curves for presentation.
    """
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    
    recalls = [r['recall_at_k'] for r in results]
    qps = [r['qps'] for r in results]
    num_tables = [r['num_tables'] for r in results]
    cand_ratios = [r['candidate_ratio'] * 100 for r in results]
    
    # Color palette
    primary_color = '#38bdf8'   # Sky blue
    accent_color = '#f43f5e'    # Rose red
    highlight_color = '#4ade80' # Emerald green
    
    # ── Left Plot: Recall@10 vs QPS (Log Scale) ──────────────────────────
    ax1.plot(recalls, qps, marker='o', markersize=7, color=primary_color, 
             linewidth=2.5, linestyle='-', label='LSH Index (num_hashes=5)')
    ax1.scatter(recalls, qps, color='#60a5fa', s=60, zorder=5)
    
    # Highlight sweet spot (num_tables=16 or closest to 90%+ recall)
    sweet_spot = max([r for r in results if r['recall_at_k'] >= 0.90], key=lambda x: x['qps'], default=results[-1])
    ss_idx = num_tables.index(sweet_spot['num_tables'])
    
    ax1.scatter([recalls[ss_idx]], [qps[ss_idx]], color=highlight_color, s=160, zorder=10, 
                edgecolors='#ffffff', linewidth=2, label=f"Sweet Spot (L={sweet_spot['num_tables']}, {recalls[ss_idx]*100:.1f}%)")
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Recall@10 (Accuracy)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Queries Per Second (QPS)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_title('Speed vs. Accuracy Tradeoff', fontsize=14, fontweight='bold', pad=12, color='#f8fafc')
    ax1.grid(True, which="both", ls="--")
    ax1.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', fontsize=10)
    
    # Annotate points
    for i, txt in enumerate(num_tables):
        if txt in [1, 3, 8, 16, 30, 50]:
            ax1.annotate(f"L={txt}", (recalls[i], qps[i]), xytext=(-15, 10), 
                         textcoords='offset points', fontsize=9, fontweight='bold', 
                         color='#e2e8f0',
                         bbox=dict(boxstyle="round,pad=0.2", fc="#0f172a", ec="#475569", alpha=0.8))

    # Callout text box for Sweet Spot
    ax1.text(0.05, 0.08, 
             f"Optimal Setting: L={sweet_spot['num_tables']}\n"
             f"Recall@10: {recalls[ss_idx]*100:.1f}%\n"
             f"Speed: {qps[ss_idx]:.0f} QPS", 
             transform=ax1.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", fc="#064e3b", ec=highlight_color, alpha=0.9), color='#f0fdf4')

    # ── Right Plot: Recall@10 & Candidate Ratio vs num_tables ─────────────
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(num_tables, [r * 100 for r in recalls], marker='s', markersize=7, 
                     color=highlight_color, linewidth=2.5, label='Recall@10 (%)')
    line2 = ax2_twin.plot(num_tables, cand_ratios, marker='^', markersize=7, 
                          color=accent_color, linewidth=2, linestyle='--', label='Candidates Examined (%)')
    
    ax2.set_xlabel('Number of Hash Tables (num_tables)', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Recall@10 (%)', fontsize=12, fontweight='bold', color=highlight_color, labelpad=10)
    ax2_twin.set_ylabel('Candidates Searched (%)', fontsize=12, fontweight='bold', color=accent_color, labelpad=10)
    ax2.set_title('Recall & Candidate Ratio vs. Hash Tables', fontsize=14, fontweight='bold', pad=12, color='#f8fafc')
    ax2.grid(True)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right', facecolor='#1e293b', edgecolor='#475569', fontsize=10)
    
    fig.suptitle('LSH Vector Database Benchmark (50,000 Vectors, Dim=128, K=10)', 
                 fontsize=16, fontweight='bold', color='#ffffff', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved publication-quality tradeoff plot to {save_path}")

def plot_candidates(results: list[dict], save_path: str = 'results/candidates_curve.png') -> None:
    """
    Plot candidate_ratio vs recall to show search space reduction efficiency.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    recalls = [r['recall_at_k'] * 100 for r in results]
    candidates = [r['candidate_ratio'] * 100 for r in results]
    num_tables = [r['num_tables'] for r in results]
    
    ax.plot(recalls, candidates, marker='o', markersize=8, color='#a855f7', linewidth=2.5, linestyle='-')
    ax.fill_between(recalls, candidates, color='#a855f7', alpha=0.15)
    
    ax.set_xlabel('Recall@10 (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Search Space Examined (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Efficiency Curve: Search Space Pruning vs. Accuracy', fontsize=14, fontweight='bold', pad=12)
    ax.grid(True)
    
    for i, txt in enumerate(num_tables):
        if txt in [1, 5, 16, 30, 50]:
            ax.annotate(f"L={txt}\n({candidates[i]:.1f}% searched)", (recalls[i], candidates[i]), 
                        xytext=(10, -15), textcoords='offset points', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="#1e293b", ec="#a855f7", alpha=0.9), color='#f3e8ff')
            
    # Reference 100% line (Brute Force)
    ax.axhline(y=100, color='#f43f5e', linestyle=':', linewidth=1.5, label='Brute Force (100% examined)')
    ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='#475569', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved publication-quality candidates curve to {save_path}")

def print_summary(results: list[dict]) -> None:
    """
    Print a clean text summary of the benchmarking results.
    """
    if not results:
        print("No results to summarize.")
        return
        
    best_recall = max(results, key=lambda x: x['recall_at_k'])
    best_qps = max(results, key=lambda x: x['qps'])
    
    sweet_spots = [r for r in results if r['recall_at_k'] >= 0.90]
    
    print("\n" + "="*60)
    print("                BENCHMARK PERFORMANCE SUMMARY")
    print("="*60)
    print(f"  • Max Accuracy:   {best_recall['recall_at_k']*100:.2f}% Recall@10 (num_tables={best_recall['num_tables']})")
    print(f"  • Peak Speed:     {best_qps['qps']:.0f} QPS (num_tables={best_qps['num_tables']})")
    
    if sweet_spots:
        sweet_spot = max(sweet_spots, key=lambda x: x['qps'])
        print(f"  • Optimal Spot:   {sweet_spot['recall_at_k']*100:.1f}% Recall@10 at {sweet_spot['qps']:.0f} QPS (num_tables={sweet_spot['num_tables']})")
        print(f"  • Efficiency:     Examined only {sweet_spot['candidate_ratio']*100:.1f}% of vectors!")
    print("="*60 + "\n")
