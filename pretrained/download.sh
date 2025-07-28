# Get the directory of the current script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

wget -P "$DIR" https://huggingface.co/cjd24/MonSter/resolve/main/mix_all.pth
wget -P "$DIR" https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
