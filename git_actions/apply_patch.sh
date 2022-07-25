#! /bin/sh

# The input should be the location of the patch file
patch_file=$2

# Save the full path of the patch file
patch_file_full_path="$(cd "$(dirname -- "$patch_file")" >/dev/null; pwd -P)/$(basename -- "$patch_file")"

filename=$(basename -- "$patch_file")
extension="${filename##*.}"

# This now stores the commit hash onto which the patch should be applied
git_commit_hash="${filename%.*}" 


# This is the spec git home
cd $1 &&\
git checkout $git_commit_hash &&\
git apply $patch_file_full_path &&\

echo "Applied the patch $patch_file_full_path on the commit $git_commit_hash in the folder $1"

# This generates a patch file whose name is the commit one should apply the patch to
# git diff $commit_hash > $commit_hash.patch