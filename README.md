\# Cow PLF



Precision Livestock Farming computer-vision platform for individual-cow monitoring.



\## Current Vision Stack



\- YOLO26m-seg

\- ByteTrack

\- V5 Permanent Cow Identity Manager

\- Appearance-based re-identification

\- Global cross-day Cow Identity Registry



\## Current Analytics



\- Instance segmentation

\- Individual-cow tracking

\- Persistent local Cow IDs

\- Persistent Global Cow IDs

\- Contactless image-space morphometrics

\- CSV measurement generation



\## Planned Analytics



\- Camera calibration

\- Real-world morphometrics

\- Contactless BCS estimation

\- Weight estimation

\- Behavioral ethograms

\- Welfare monitoring

\- Facial/emotion indicators

\- Aggression detection

\- Social dynamics

\- Flow analytics

\- High-density tracking



\## Running



Activate the environment and run:



```powershell

cd D:\\cow

$env:PYTHONPATH="D:\\cow\\src"

py -3.12 main.py --video videos\\cow\_video6.mp4

