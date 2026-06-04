#!/usr/bin/env bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 --wav_dir /path/to/wav_folder --txt_dir /path/to/txt_folder [--train_ratio 0.8 --dev_ratio 0.1]
Options:
    --wav_dir: wav 檔資料夾
    --txt_dir: txt 檔資料夾
    --train_ratio: 訓練集比例 (default=0.8)
    --dev_ratio: 開發集比例 (default=0.1，剩下為 test)
EOF
)

SECONDS=0
train_ratio=0.8
dev_ratio=0.1
wav_dir=
txt_dir=
. ./utils/parse_options.sh

if [ -z "${wav_dir}" ] || [ -z "${txt_dir}" ]; then
    log "Error: 必須同時指定 --wav_dir 和 --txt_dir"
    echo "${help_message}"
    exit 1
fi

if [ ! -d "${wav_dir}" ] || [ ! -d "${txt_dir}" ]; then
    log "Error: wav_dir 和 txt_dir 必須是存在的資料夾"
    exit 1
fi

log "wav 來源: ${wav_dir}"
log "txt 來源: ${txt_dir}"

local_dir=data/local
mkdir -p ${local_dir}/{train,dev,test}

# 1. 收集所有 wav 檔
find "${wav_dir}" -type f -iname "*.wav" | sort > ${local_dir}/all_wav.scp
n=$(wc -l < ${local_dir}/all_wav.scp)
log "共找到 ${n} 個 wav 檔"

# 2. 檢查 txt 是否一一對應，缺失則過濾
missing_txt=0
> ${local_dir}/all_wav_filtered.scp
while read -r wav_path; do
    utt=$(basename "$wav_path" .wav)
    if [ ! -f "${txt_dir}/${utt}.txt" ]; then
        log "缺少對應的 txt: ${txt_dir}/${utt}.txt"
        missing_txt=$((missing_txt+1))
    else
        echo "$wav_path" >> ${local_dir}/all_wav_filtered.scp
    fi
done < ${local_dir}/all_wav.scp

mv ${local_dir}/all_wav_filtered.scp ${local_dir}/all_wav.scp
n=$(wc -l < ${local_dir}/all_wav.scp)
log "檢查後剩餘 $n 個有效 wav 檔（缺失 $missing_txt 個）"



# 3. 隨機打亂並分割 train/dev/test
# 3. 按 speaker 切分資料
> ${local_dir}/train/wav.flist
> ${local_dir}/dev/wav.flist
> ${local_dir}/test/wav.flist

# # 先處理要全部放到 test 的 speaker (01, 25)
# grep -E "/S0(01|25)" ${local_dir}/all_wav.scp >> ${local_dir}/test/wav.flist

# # 找出剩餘 speaker 列表
# speakers=$(awk -F'/' '{fname=$NF; spkid=substr(fname,3,2); print spkid}' ${local_dir}/all_wav.scp | sort -u | grep -Ev "^(01|25)$")

# for spk in $speakers; do
#     spk_wavs=$(grep -E "/S0${spk}" ${local_dir}/all_wav.scp | shuf)
#     n_spk=$(echo "$spk_wavs" | wc -l)
#     n_train=$(python3 -c "print(int(${n_spk} * ${train_ratio}))")
#     n_dev=$(python3 -c "print(int(${n_spk} * ${dev_ratio}))")
#     n_test=$((n_spk - n_train - n_dev))

#     echo "$spk_wavs" | sed -n "1,${n_train}p" >> ${local_dir}/train/wav.flist
#     echo "$spk_wavs" | sed -n "$((n_train + 1)),$((n_train + n_dev))p" >> ${local_dir}/dev/wav.flist
#     echo "$spk_wavs" | sed -n "$((n_train + n_dev + 1)),$((n_train + n_dev + n_test))p" >> ${local_dir}/test/wav.flist
# done
# 指定四個 test speaker
test_speakers="01|03|10|25"

# 收集全部 test speaker 檔案
grep -E "/S0($test_speakers)" ${local_dir}/all_wav.scp >> ${local_dir}/test/wav.flist

# 取得剩下 speaker id (去除上述四個)
speakers=$(awk -F'/' '{fname=$NF; spkid=substr(fname,3,2); print spkid}' ${local_dir}/all_wav.scp | sort -u | grep -Ev "^($(echo $test_speakers | sed 's/|/|/g'))$")

for spk in $speakers; do
    spk_wavs=$(grep -E "/S0${spk}" ${local_dir}/all_wav.scp | shuf)
    n_spk=$(echo "$spk_wavs" | wc -l)
    n_train=$(python3 -c "print(int(${n_spk} * ${train_ratio}))")
    n_dev=$(python3 -c "print(int(${n_spk} * ${dev_ratio}))")
    n_test=$((n_spk - n_train - n_dev))

    echo "$spk_wavs" | sed -n "1,${n_train}p" >> ${local_dir}/train/wav.flist
    echo "$spk_wavs" | sed -n "$((n_train + 1)),$((n_train + n_dev))p" >> ${local_dir}/dev/wav.flist
    echo "$spk_wavs" | sed -n "$((n_train + n_dev + 1)),$((n_train + n_dev + n_test))p" >> ${local_dir}/test/wav.flist
done

log "train/dev/test split 完成："
log "Train: $(wc -l < ${local_dir}/train/wav.flist)"
log "Dev: $(wc -l < ${local_dir}/dev/wav.flist)"
log "Test: $(wc -l < ${local_dir}/test/wav.flist)"



# 4. 產生 utt.list
for split in train dev test; do
    sed -e 's/\.wav$//' ${local_dir}/${split}/wav.flist | awk -F '/' '{print $NF}' > ${local_dir}/${split}/utt.list
done
log "utt_list successs"
# 5. 產生 utt2spk_all / wav.scp_all / transcripts_all
for split in train dev test; do
    awk '{utt=$1; spk=substr(utt,3,2); print utt, spk}' ${local_dir}/${split}/utt.list > ${local_dir}/${split}/utt2spk_all
    paste -d' ' ${local_dir}/${split}/utt.list ${local_dir}/${split}/wav.flist > ${local_dir}/${split}/wav.scp_all
    while read -r utt; do
        txt_file="${txt_dir}/${utt}.txt"
        text=$(cat "$txt_file" | tr -d "[:punct:]" | tr -d "\r")
        echo "$utt $text"
    done < ${local_dir}/${split}/utt.list > ${local_dir}/${split}/transcripts_all
done
log "spklist success"
# 6. 過濾 & 排序
for split in train dev test; do
    utils/filter_scp.pl -f 1 ${local_dir}/${split}/utt.list ${local_dir}/${split}/utt2spk_all | sort -u > ${local_dir}/${split}/utt2spk
    utils/filter_scp.pl -f 1 ${local_dir}/${split}/utt.list ${local_dir}/${split}/wav.scp_all | sort -u > ${local_dir}/${split}/wav.scp
    utils/filter_scp.pl -f 1 ${local_dir}/${split}/utt.list ${local_dir}/${split}/transcripts_all | sort -u > ${local_dir}/${split}/text
    utils/utt2spk_to_spk2utt.pl ${local_dir}/${split}/utt2spk > ${local_dir}/${split}/spk2utt
done
log "filter"
# 7. 複製到 data/{train,dev,test}
mkdir -p data/{train,dev,test}
for split in train dev test; do
    cp ${local_dir}/${split}/{spk2utt,utt2spk,wav.scp,text} data/${split}/
done

# 8. 去掉 text 中多餘空格
for x in train dev test; do
    cp data/${x}/text data/${x}/text.org
    paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") > data/${x}/text
    rm data/${x}/text.org
done

log "資料準備完成 [elapsed=${SECONDS}s]"
