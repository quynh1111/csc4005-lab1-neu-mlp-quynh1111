from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.dataset import create_dataloaders
from src.model import MLPClassifier
from src.utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export correct/incorrect prediction examples from best model')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 64])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--num_correct', type=int, default=6)
    parser.add_argument('--num_wrong', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()


def predict_label(model: MLPClassifier, image_path: Path, transform, device: torch.device) -> int:
    image = Image.open(image_path).convert('L')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()
    return pred


def render_examples(samples: list[tuple[Path, str, str]], title: str, output_path: Path) -> None:
    if not samples:
        return

    n = len(samples)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))

    if hasattr(axes, 'flat'):
        axes_list = list(axes.flat)
    else:
        axes_list = [axes]

    for ax in axes_list:
        ax.axis('off')

    for idx, (path, true_name, pred_name) in enumerate(samples):
        ax = axes_list[idx]
        image = Image.open(path).convert('L')
        ax.imshow(image, cmap='gray')
        ax.set_title(f'True: {true_name}\nPred: {pred_name}', fontsize=10)
        ax.axis('off')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = ensure_dir(Path('outputs') / args.run_name)

    data = create_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        augment=False,
        num_workers=args.num_workers,
    )

    model = MLPClassifier(
        input_dim=data.input_dim,
        num_classes=len(data.class_names),
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    best_model_path = output_dir / 'best_model.pt'
    if not best_model_path.exists():
        raise FileNotFoundError(f'Cannot find best model at: {best_model_path}')

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    class_names = data.class_names
    test_ds = data.test_loader.dataset

    correct_examples: list[tuple[Path, str, str]] = []
    wrong_examples: list[tuple[Path, str, str]] = []

    for image_path, true_label in test_ds.samples:
        pred_label = predict_label(model, image_path, test_ds.transform, device)
        true_name = class_names[true_label]
        pred_name = class_names[pred_label]

        if pred_label == true_label and len(correct_examples) < args.num_correct:
            correct_examples.append((image_path, true_name, pred_name))
        if pred_label != true_label and len(wrong_examples) < args.num_wrong:
            wrong_examples.append((image_path, true_name, pred_name))

        if len(correct_examples) >= args.num_correct and len(wrong_examples) >= args.num_wrong:
            break

    render_examples(
        correct_examples,
        title=f'Correct Predictions - {args.run_name}',
        output_path=output_dir / 'examples_correct.png',
    )
    render_examples(
        wrong_examples,
        title=f'Incorrect Predictions - {args.run_name}',
        output_path=output_dir / 'examples_wrong.png',
    )

    print(f'Saved: {output_dir / "examples_correct.png"}')
    print(f'Saved: {output_dir / "examples_wrong.png"}')


if __name__ == '__main__':
    main()
