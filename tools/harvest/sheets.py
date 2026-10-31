"""Контактные листы для ручного аудита: ≤ 8×8 лиц на изображение, подписи id/метка/sim."""
from __future__ import annotations

TILE, CAPTION, COLS, ROWS = 144, 30, 8, 8
COLORS = {('accepted', '1'): (40, 160, 70), ('accepted', '0'): (50, 110, 200),
          ('ambiguous', '1'): (235, 150, 20), ('ambiguous', '0'): (235, 150, 20)}


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 без FreeType
        return ImageFont.load_default()


def sort_rows(rows, order):
    if order == 'sim':
        return sorted(rows, key=lambda r: float(r['sokolov_similarity'] or 0), reverse=True)
    return sorted(rows, key=lambda r: (r['source_id'], float(r['timestamp_seconds'] or 0), r['image_id']))


def render(root, rows, destination, title):
    """Пишет destination(.jpg) и при > 64 лицах destination_pNN.jpg. Возвращает пути."""
    from PIL import Image, ImageDraw
    per_page = COLS * ROWS
    pages = [rows[i:i + per_page] for i in range(0, len(rows), per_page)] or [[]]
    font, small = _font(13), _font(12)
    written = []
    for number, page in enumerate(pages, 1):
        lines = (len(page) + COLS - 1) // COLS or 1
        sheet = Image.new('RGB', (COLS * TILE, 26 + lines * (TILE + CAPTION)), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 6), f'{title}  [{number}/{len(pages)}]  green=1 accepted, blue=0 accepted, '
                          f'orange=ambiguous, grey=excluded', fill='black', font=small)
        for i, row in enumerate(page):
            x, y = (i % COLS) * TILE, 26 + (i // COLS) * (TILE + CAPTION)
            try:
                with Image.open(root / row['path']) as face:
                    tile = face.convert('RGB').resize((TILE - 6, TILE - 6))
            except OSError:
                tile = Image.new('RGB', (TILE - 6, TILE - 6), (90, 0, 0))
            color = COLORS.get((row['review_status'], row['label']), (150, 150, 150))
            draw.rectangle([x, y, x + TILE - 1, y + TILE - 1], fill=color)
            sheet.paste(tile, (x + 3, y + 3))
            suffix = row['image_id'].rsplit('_', 1)[-1]
            ts = f' t={float(row["timestamp_seconds"]):.0f}' if row['timestamp_seconds'] else ''
            draw.text((x + 3, y + TILE), f'{suffix} L{row["label"]} {row["review_status"][:3]}{ts}',
                      fill='black', font=small)
            draw.text((x + 3, y + TILE + 14), f'sim {float(row["sokolov_similarity"]):+.3f} {row["person"][:10]}',
                      fill='black', font=font)
        path = destination if number == 1 else destination.with_name(f'{destination.stem}_p{number:02d}.jpg')
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, quality=90)
        written.append(path)
    stale = number + 1
    while (old := destination.with_name(f'{destination.stem}_p{stale:02d}.jpg')).exists():
        old.unlink()
        stale += 1
    return written
