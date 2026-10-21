// Vercel: exit 0 skips data-only commits; exit 1 builds code/preview branches.
const branch=process.env.VERCEL_GIT_COMMIT_REF;
const data=process.env.LABEL_DATA_BRANCH || 'RADU/labeling-data';
process.exit(branch===data ? 0 : 1);
