# The input should be some commit in the main spec git repo
commit_hash=$1

# This generates a patch file whose name is the commit one should apply the patch to
git diff $commit_hash > $commit_hash.patch