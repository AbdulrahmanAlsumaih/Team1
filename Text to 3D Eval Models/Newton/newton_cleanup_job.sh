#!/bin/bash
#SBATCH --job-name=cleanup_job
#SBATCH --output=job_output_%j.log
#SBATCH --error=job_error_%j.log
#SBATCH --time=5-00:00:00
#SBATCH -c 1                # Request 1 CPU
#SBATCH --mem=1G            # Allocate 1 GB of memory
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa181349@ucf.edu

# Run the cleanup command every 30 minutes for up to 5 days
end_time=$(( $(date +%s) + 5 * 24 * 60 * 60 ))  # 5 days in seconds

while [[ $(date +%s) -lt $end_time ]]; do
    find /home/cap6411.student2/cvs/threestudio/outputs -type f ! -name "*.mp4" -mmin +30 -exec rm -f {} +
    sleep 1800  # Wait for 30 minutes (1800 seconds) before running again
done
