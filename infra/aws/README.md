# Deploying to AWS

The API is a single container. The simplest path is **Amazon ECR + AWS App Runner**
(or ECS Fargate if you need more control). Both run on CPU; a GPU instance (EC2 `g5.xlarge`
or SageMaker) is only needed for faster generation or for fine-tuning.

## 1. Build the index and adapter locally

```bash
make data index          # Synthea + CMS docs -> data/index, data/processed
make finetune-data train # optional: LoRA adapter -> models/lora-adapter (GPU recommended)
```

Copy `data/index`, `data/processed` and `models/` into the image, or upload them to S3
and download them at container start.

## 2. Push the image to ECR

```bash
AWS_REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO=clinical-insight-assistant

aws ecr create-repository --repository-name $REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t $REPO .
docker tag $REPO:latest $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
```

## 3. Run it on App Runner

In the App Runner console: **Create service -> Container registry -> Amazon ECR**, pick the
image, set port `8000`, health check path `/health`, and at least **2 vCPU / 4 GB memory**
(the 1.5B-parameter model needs about 3 GB in float32).

## Cost and security notes

- Stop or pause the service when you're not demoing it.
- Keep the service private (VPC ingress) or put it behind API Gateway with an API key.
- The data is synthetic, but treat the deployment as if it held PHI: encryption at rest,
  no request logging of question text, least-privilege IAM roles.
