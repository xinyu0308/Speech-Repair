#!/bin/bash

# Define the directories
txt_dir="/home/xinyu/xinyu/Desktop/CDSD_Text"
wav_dir="/home/xinyu/xinyu/Desktop/CDSD_0811"

# Output files
text_output="text"
wav_scp_output="wav.scp"

# Empty the output files if they already exist
> "$text_output"
> "$wav_scp_output"

# Generate the text file
echo "Generating $text_output..."
find "$txt_dir" -type f -iname "*.txt" | while read -r txt_file; do
    tag=$(basename "$txt_file" .txt)
    # Extract only Chinese characters
    content=$(grep -oP "[\x{4e00}-\x{9fa5}]" "$txt_file" | tr -d '\n')
    
    # Count the number of Chinese characters
    char_count=$(echo -n "$content" | wc -m)

    # Check if character count is more than 3
    if [ "$char_count" -gt 0 ]; then
        # Separate each character with a space
        spaced_content=$(echo "$content" | sed 's/./& /g')
        echo "$tag $spaced_content" >> "$text_output"
    else
        echo "Skipping $tag, not enough characters."
    fi
done

# Generate the wav_scp file
echo "Generating $wav_scp_output..."
while IFS=' ' read -r tag spaced_text; do
    wav_file=$(find "$wav_dir" -type f -name "${tag}.wav")
    if [ -f "$wav_file" ]; then
        echo "$tag $wav_file" >> "$wav_scp_output"
    else
        echo "No matching .wav file found for tag: $tag"
    fi
done < "$text_output"
# echo "Generating $text_output and $wav_scp_output..."
# find "$txt_dir" -type f -iname "*.txt" | while read -r txt_file; do
#     tag=$(basename "$txt_file" .txt)
#     wav_file=$(find "$wav_dir" -type f -name "${tag}.wav")
    
#     # Only process if the .wav file exists
#     if [ -f "$wav_file" ]; then
#         # Generate text file
#         content=$(grep -oP "[\x{4e00}-\x{9fa5}]" "$txt_file" | tr -d '\n')
#         char_count=$(echo -n "$content" | wc -m)
#         if [ "$char_count" -gt 1 ]; then
#             spaced_content=$(echo "$content" | sed 's/./& /g')
#             echo "$tag $spaced_content" >> "$text_output"
            
#             # Generate wav.scp file
#             echo "$tag $wav_file" >> "$wav_scp_output"
#         else
#             echo "Skipping $tag, not enough characters."
#         fi
#     else
#         echo "No matching .wav file found for tag: $tag"
#     fi
# done
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Data preparation related
remove_archive=false
download_opt=

log "$0 $*"

. ./utils/parse_options.sh
. ./db.sh
. ./path.sh
. ./cmd.sh

aishell_text="$text_output"
input_wav_scp="$wav_scp_output"

# Data Preparation
train_dir=data/local/train
dev_dir=data/local/dev
test_dir=data/local/test
tmp_dir=data/local/tmp

mkdir -p "$train_dir"
mkdir -p "$dev_dir"
mkdir -p "$test_dir"
mkdir -p "$tmp_dir"

# Copying the wav.scp file to the temporary directory
cp "$input_wav_scp" "$tmp_dir/wav.scp"

# Define ratios
train_ratio=0.8
dev_ratio=0.1
test_ratio=0.1

# Randomly shuffle the data
shuf "$tmp_dir/wav.scp" > "$tmp_dir/wav.scp.shuffled"

# Count the total number of lines
total_lines=$(wc -l < "$tmp_dir/wav.scp.shuffled")

# Calculate the number of lines for each set
train_lines=$(echo "$total_lines * $train_ratio" | bc | cut -f1 -d".")
dev_lines=$(echo "$total_lines * $dev_ratio" | bc | cut -f1 -d".")
test_lines=$(echo "$total_lines - $train_lines - $dev_lines" | bc)

# Split the data into train, dev, and test sets
head -n "$train_lines" "$tmp_dir/wav.scp.shuffled" > "$train_dir/wav.scp"
tail -n "+$((train_lines + 1))" "$tmp_dir/wav.scp.shuffled" | head -n "$dev_lines" > "$dev_dir/wav.scp"
tail -n "+$((train_lines + dev_lines + 1))" "$tmp_dir/wav.scp.shuffled" > "$test_dir/wav.scp"

# Clean up temporary files
rm -r "$tmp_dir"

# Transcriptions preparation
for dir in $train_dir $dev_dir $test_dir; do
  log Preparing $dir transcriptions
  sed -e 's/\.wav//' $dir/wav.scp | awk -F '/' '{print $NF}' > $dir/utt.list
  sed -e 's/\.wav//' $dir/wav.scp | awk -F '/' '{i=NF-1;printf("%s %s\n",$NF,$i)}' > $dir/utt2spk_all
  paste $dir/wav.scp > $dir/wav.scp_all
  utils/filter_scp.pl -f 1 $dir/utt.list $aishell_text > $dir/transcripts.txt
  awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/utt2spk_all | sort -u > $dir/utt2spk
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
  sort -u $dir/transcripts.txt > $dir/text
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

mkdir -p data/train data/dev data/test

for f in spk2utt utt2spk wav.scp text; do
  cp "$train_dir/$f" "data/train/$f" || exit 1;
  cp "$dev_dir/$f" "data/dev/$f" || exit 1;
  cp "$test_dir/$f" "data/test/$f" || exit 1;
done

# remove space in text
for x in train dev test; do
  cp "data/${x}/text" "data/${x}/text.org"
  paste -d " " <(cut -f 1 -d" " "data/${x}/text.org") <(cut -f 2- -d" " "data/${x}/text.org" | tr -d " ") \
      > "data/${x}/text"
  rm "data/${x}/text.org"
done

log "Successfully finished."
