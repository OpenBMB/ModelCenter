# python /mnt/sfs_turbo/hx/CPM-2.1/src/tools/preprocess_data_enc_dec_lm.py
# python /mnt/sfs_turbo/hx/CPM-2.1/src/tools/preprocess_data_enc_dec_mlm.py
# python /mnt/sfs_turbo/hx/CPM-2.1/src/tools/preprocess_data_enc_dec.pys

for ((i=$1; i<$2; i++)); do
{
    python3 /mnt/sfs_turbo/hx/cpm3-pretrain/src/tools/preprocess_cpm1_lm_new.py --uid $i
}
done

