# Detecția stării de somnolență a conducătorilor auto (UTA-RLDD)

Proiect de laborator CV 3: clasificarea cadrelor video în 3 clase: **alert (0)**, **vigilență scăzută (5)**, **somnolență (10)**.

> Stadiu actual: încărcarea și pregătirea datelor (extragere cadre, split train/val/test, `Dataset`, `DataLoader`).
> Modelul, antrenarea, testarea, graficele și augmentarea generativă vor fi adăugate în commit-urile următoare.

## 1. Mediu și instalare

Python 3.11 (testat cu 3.11.9), într-un mediu virtual dedicat. Rulează pe CPU.

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Versiunile pachetelor sunt fixate în `requirements.txt`.

## 2. Descărcarea setului de date

Se folosește **UTA-RLDD** (University of Texas at Arlington Real-Life Drowsiness Dataset): 60 de participanți, 3 videoclipuri de ~10 minute pentru fiecare (alert, vigilență scăzută, somnolență).

Pagina oficială (cu link de descărcare): <https://sites.google.com/view/utarldd/home>

Datele sunt împărțite în foldere. Am folosit **Fold3_part1 și Fold3_part2** (participanții 25–36, 12 persoane, 37 de videoclipuri), din cauza resurselor limitate (CPU, fără GPU).

Dezarhivează-le în `data/raw/`. Structura așteptată:

```
data/raw/Fold3_part1/Fold3_part1/25/0.mp4
data/raw/Fold3_part1/Fold3_part1/25/5.mp4
data/raw/Fold3_part1/Fold3_part1/25/10.mp4
...
data/raw/Fold3_part2/Fold3_part2/32/10_1.mp4   (clasa 10 poate fi împărțită în mai multe fișiere)
```

Eticheta se citește din numele fișierului (`0`, `5`, `10`, sau `10_1`, `10_2` etc.), iar participantul din numele folderului. Extensiile acceptate: `.mp4 .mov .avi .mkv .m4v`.
Datele **nu se urcă în git** (sunt în `.gitignore`).

## 3. Pregătirea datelor

```bash
python -m src.prepare_data
```

Ce face:
1. Extrage **un cadru la fiecare 2 secunde** din fiecare video, îl redimensionează (latura scurtă la 160 px), îl decupează central la 160×160 și îl salvează JPEG în `data/processed/frames/`.
2. Împarte cadrele în **train / val / test** (70% / 15% / 15% nominal, seed 42) și scrie `data/processed/splits.csv`.
3. Afișează o verificare a împărțirii.

Cu cele 12 persoane, rezultă ~10.000 de cadre (train 7.559, val 1.216, test 1.184) în aproximativ 16 minute, pe un procesor AMD Ryzen 5 5500U. Extragerea poate fi reluată: videoclipurile terminate sunt sărite (manifeste în `data/processed/manifests/`).

Opțiuni utile: `--every-sec`, `--size`, `--workers`, `--seed`, `--limit-videos N` (test rapid), `--mode`.

### Strategia de împărțire (detecție dependentă de subiect)

Cadrele vecine din același video sunt aproape identice, deci o împărțire aleatorie pe cadre ar plasa „gemeni” în train și în test. De aceea:

- fiecare video este tăiat în **segmente de 30 s**, iar segmentele (nu cadrele) sunt repartizate aleatoriu în train/val/test, separat pentru fiecare video;
- fiecare participant și fiecare clasă apar în toate cele 3 split-uri;
- la granița dintre segmente din split-uri diferite se elimină cadrele aflate la mai puțin de **5 s** de graniță (separare minimă de 10 s între split-uri). Prin eliminarea lor, proporțiile reale devin ~76% / 12% / 12%.

**Limită:** cadrele de test provin din aceleași videoclipuri (deci aceiași participanți, fundaluri și condiții de lumină) ca cele de train, deci rezultatele măsoară recunoașterea în interiorul acelorași persoane, nu generalizarea la șoferi noi. Opțiunea `--mode subject` repartizează participanți întregi în split-uri (detecție independentă de subiect) și va fi folosită pentru comparație.

## 4. Verificare

```bash
python -m src.inspect_data
```

Afișează numărul de cadre pe split și clasă, încarcă un batch din `DataLoader` și salvează în `outputs/` imaginile `samples.png` (cadre exemplu pe clase) și `split_timeline.png` (repartizarea segmentelor în câteva videoclipuri).

## 5. Structura proiectului

```
src/
  config.py            etichete, nume de clase, seed
  prepare_data.py      video -> cadre -> splits.csv
  inspect_data.py      statistici, grafice, test DataLoader
  data/
    extract_frames.py  găsirea videoclipurilor, extragerea și redimensionarea cadrelor
    splits.py          împărțire train/val/test și verificări
    dataset.py         DrowsinessDataset, transformări, get_dataloaders
```

## 6. Dataset și DataLoader

- `DrowsinessDataset(frames, root, split, transform)` returnează `(imagine, clasă)`, cu clasele `0 = alert`, `1 = low_vigilance`, `2 = drowsy`.
- `build_transforms(split, image_size=128)`: pe **train** decupare aleatorie, flip orizontal, rotație ±5°, variații de luminozitate și contrast; pe **val/test** doar redimensionare. În ambele cazuri urmează `ToTensor` și normalizare cu statisticile ImageNet.
- `get_dataloaders(...)` creează 3 `DataLoader`-e; doar train este amestecat, cu generator și workeri seedați pentru reproductibilitate. Pe Windows, cu `num_workers > 0`, scriptul apelant trebuie să aibă `if __name__ == "__main__":`.