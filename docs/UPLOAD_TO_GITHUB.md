# Upload the reviewed code package

This package has not been uploaded. The GitHub plugin was declined, and no destination repository or authenticated upload session was available during preparation.

Before making a public release, the copyright holder must choose a licence and replace `LICENSE_PENDING.txt` with the corresponding `LICENSE`. Complete the author metadata in `CITATION.cff.template`, rename it to `CITATION.cff`, and add the actual repository URL and release version to the README and manuscript. Do not assert a DOI that has not been issued.

## Browser upload

Create or open the intended repository on GitHub, then use **Add file → Upload files**. Upload the extracted contents of `RCCR-MoE`, keeping its directory structure. Uploading the ZIP alone does not create a browsable source tree. Include `.gitignore`; omit all generated data and result folders.

## Existing Git client

For a new empty remote repository, start from the extracted `RCCR-MoE` folder:

```bash
git init -b main
git add .
git status --short
git commit -m "Prepare RCCR-MoE research code"
git remote add origin https://github.com/OWNER/RCCR-MoE.git
git push -u origin main
```

Replace `OWNER` and the repository name with the actual destination. Authenticate through the client's normal workflow. Never put tokens into the repository or a command history. If the remote already contains work, use a clone and a review branch instead of forcing a push.

After upload, verify that the remote contains only the intended files, that the README links work, and that the citation and licence are accurate. Record the resulting commit and release tag in the manuscript. The current review package makes no claim that these steps have been completed.
