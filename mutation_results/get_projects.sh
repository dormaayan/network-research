#!/bin/bash
set -e
cd /home/dorma10/lightweight-effectiveness/projects
git clone https://github.com/eBay/cors-filter.git cors-filter
cd /home/dorma10/lightweight-effectiveness/projects/cors-filter
git fetch --tags
latestTag=$(git describe --tags `git rev-list --tags --max-count=1`)
git checkout $latestTag
mvn clean install -DskipTests
mvn test -Dmaven.test.failure.ignore=true
cd ..
git clone https://github.com/square/retrofit.git retrofit
cd /home/dorma10/lightweight-effectiveness/projects/retrofit
git fetch --tags
latestTag=$(git describe --tags `git rev-list --tags --max-count=1`)
git checkout $latestTag
mvn clean install -DskipTests
mvn test -Dmaven.test.failure.ignore=true
cd ..
cd ..
