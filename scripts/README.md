# scripts

## Terraform

### Local state files left by `bootstrap.sh`

`bootstrap.sh` keeps the bootstrap's Terraform state in the S3 state bucket
(`bootstrap/terraform.tfstate`). If `terraform/bootstrap` already has a local
`terraform.tfstate` when it runs, the script never applies that file to the wrong
account:

- **`stale-<timestamp>.tfstate`**: state for a different AWS account than the one
  you're signed in to, or a file with no resources. It's renamed and left alone.
- **`migrated-<timestamp>.tfstate`** and **`migrated-<timestamp>.backup.tfstate`**:
  state for this account, after the script has moved it into S3. Terraform empties
  the local file once the move is done; the script renames it and Terraform's
  backup so the next run doesn't mistake them for state still waiting to move.

Neither is default Terraform behaviour: `bootstrap.sh` names them. Both end in
`.tfstate`, so the repo's `.gitignore` keeps them out of git.

Delete the `migrated-` files whenever you like; S3 holds the real copy and keeps
every version. Treat a `stale-` file as a credential: Terraform keeps some secrets
(such as IAM access keys) in state in plain text. Delete it once you no longer
need it.
