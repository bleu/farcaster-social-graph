### Start LocalStack (in a separate terminal):

```
docker run --rm -it -p 4566:4566 -p 4571:4571 localstack/localstack:latest
```

### Run cdk commands with the local version:

```
npm run cdklocal bootstrap
npm run cdklocal deploy
```
