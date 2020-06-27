#!/bin/bash
rm -rf /home/dorma10/lightweight-effectiveness/mutation_results logs
mkdir /home/dorma10/lightweight-effectiveness/mutation_results logs
echo '* 1 out of 4 -> empire-db'
mkdir /home/dorma10/lightweight-effectiveness/mutation_results/empire-db


echo '* Compiling empire-db'
cd /home/dorma10/lightweight-effectiveness/projects/empire-db

echo '* Caching original pom'
cp pom.xml cached_pom.xml

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.commons.Attributes org.apache.empire.commons.AttributesTest DEFAULT
echo '* Mutating org.apache.empire.commons.Attributes with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.commons.AttributesTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.commons.AttributesTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.commons.AttributesTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.commons.AttributesTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.commons.Options org.apache.empire.commons.OptionsTest DEFAULT
echo '* Mutating org.apache.empire.commons.Options with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.commons.OptionsTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.commons.OptionsTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.commons.OptionsTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.commons.OptionsTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.commons.DateUtils org.apache.empire.commons.DateUtilsTest DEFAULT
echo '* Mutating org.apache.empire.commons.DateUtils with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.commons.DateUtilsTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.commons.DateUtilsTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.commons.DateUtilsTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.commons.DateUtilsTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.commons.ErrorType org.apache.empire.commons.ErrorTypeTest DEFAULT
echo '* Mutating org.apache.empire.commons.ErrorType with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.commons.ErrorTypeTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.commons.ErrorTypeTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.commons.ErrorTypeTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.commons.ErrorTypeTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.commons.ObjectUtils org.apache.empire.commons.ObjectUtilsTest DEFAULT
echo '* Mutating org.apache.empire.commons.ObjectUtils with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.commons.ObjectUtilsTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.commons.ObjectUtilsTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.commons.ObjectUtilsTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.commons.ObjectUtilsTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.commons.StringUtils org.apache.empire.commons.StringUtilsTest DEFAULT
echo '* Mutating org.apache.empire.commons.StringUtils with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.commons.StringUtilsTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.commons.StringUtilsTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.commons.StringUtilsTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.commons.StringUtilsTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.commons.OptionEntry org.apache.empire.commons.OptionEntryTest DEFAULT
echo '* Mutating org.apache.empire.commons.OptionEntry with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.commons.OptionEntryTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.commons.OptionEntryTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.commons.OptionEntryTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.commons.OptionEntryTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.db.DBCommand org.apache.empire.db.DBCommandTest DEFAULT
echo '* Mutating org.apache.empire.db.DBCommand with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.db.DBCommandTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.db.DBCommandTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.db.DBCommandTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.db.DBCommandTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.db.expr.set.DBSetExpr org.apache.empire.db.expr.set.DBSetExprTest DEFAULT
echo '* Mutating org.apache.empire.db.expr.set.DBSetExpr with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.db.expr.set.DBSetExprTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.db.expr.set.DBSetExprTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.db.expr.set.DBSetExprTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.db.expr.set.DBSetExprTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.db.mysql.DBDatabaseDriverMySQL org.apache.empire.db.mysql.DBDatabaseDriverMySQLTest DEFAULT
echo '* Mutating org.apache.empire.db.mysql.DBDatabaseDriverMySQL with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.db.mysql.DBDatabaseDriverMySQLTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.db.mysql.DBDatabaseDriverMySQLTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.db.mysql.DBDatabaseDriverMySQLTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.db.mysql.DBDatabaseDriverMySQLTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.db.postgresql.DBDatabaseDriverPostgreSQL org.apache.empire.db.postgresql.DBDatabaseDriverPostgreSQLTest DEFAULT
echo '* Mutating org.apache.empire.db.postgresql.DBDatabaseDriverPostgreSQL with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.db.postgresql.DBDatabaseDriverPostgreSQLTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.db.postgresql.DBDatabaseDriverPostgreSQLTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.db.postgresql.DBDatabaseDriverPostgreSQLTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.db.postgresql.DBDatabaseDriverPostgreSQLTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.db.sqlite.DBDatabaseDriverSQLite org.apache.empire.db.sqlite.DBDatabaseDriverSQLiteTest DEFAULT
echo '* Mutating org.apache.empire.db.sqlite.DBDatabaseDriverSQLite with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.db.sqlite.DBDatabaseDriverSQLiteTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.db.sqlite.DBDatabaseDriverSQLiteTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.db.sqlite.DBDatabaseDriverSQLiteTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.db.sqlite.DBDatabaseDriverSQLiteTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.db.hsql.DBDatabaseDriverHSql org.apache.empire.db.hsql.DBDatabaseDriverHSqlTest DEFAULT
echo '* Mutating org.apache.empire.db.hsql.DBDatabaseDriverHSql with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-org.apache.empire.db.hsql.DBDatabaseDriverHSqlTest.txt
mv empire-db/target/pit-reports empire-db/target/empire-db-org.apache.empire.db.hsql.DBDatabaseDriverHSqlTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db/target/empire-db-org.apache.empire.db.hsql.DBDatabaseDriverHSqlTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db/target/empire-db-org.apache.empire.db.hsql.DBDatabaseDriverHSqlTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.db.codegen.CodeGenParser org.apache.empire.db.codegen.CodeGenParserTest DEFAULT
echo '* Mutating org.apache.empire.db.codegen.CodeGenParser with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-codegen-org.apache.empire.db.codegen.CodeGenParserTest.txt
mv empire-db-codegen/target/pit-reports empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.CodeGenParserTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.CodeGenParserTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.CodeGenParserTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.db.codegen.WriterService org.apache.empire.db.codegen.WriterServiceTest DEFAULT
echo '* Mutating org.apache.empire.db.codegen.WriterService with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-codegen-org.apache.empire.db.codegen.WriterServiceTest.txt
mv empire-db-codegen/target/pit-reports empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.WriterServiceTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.WriterServiceTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.WriterServiceTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.db.codegen.util.DBUtil org.apache.empire.db.codegen.util.DBUtilTest DEFAULT
echo '* Mutating org.apache.empire.db.codegen.util.DBUtil with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-codegen-org.apache.empire.db.codegen.util.DBUtilTest.txt
mv empire-db-codegen/target/pit-reports empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.util.DBUtilTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.util.DBUtilTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.util.DBUtilTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py empire-db org.apache.empire.db.codegen.util.FileUtils org.apache.empire.db.codegen.util.FileUtilsTest DEFAULT
echo '* Mutating org.apache.empire.db.codegen.util.FileUtils with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/empire-db-codegen-org.apache.empire.db.codegen.util.FileUtilsTest.txt
mv empire-db-codegen/target/pit-reports empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.util.FileUtilsTest
rm -rf empire-db/target/pit-reports
rm -rf empire-db-struts2/target/pit-reports
rm -rf empire-db-jsf2/target/pit-reports
rm -rf empire-db-codegen/target/pit-reports
rm -rf empire-db-maven-plugin/target/pit-reports
rm -rf empire-db-spring/target/pit-reports
rm -rf empire-db-examples/target/pit-reports
cp -r empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.util.FileUtilsTest /home/dorma10/lightweight-effectiveness/mutation_results/empire-db

rm -rf empire-db-codegen/target/empire-db-codegen-org.apache.empire.db.codegen.util.FileUtilsTest
echo '* Restoring original pom'
rm -rf pom.xml
mv cached_pom.xml pom.xml
cd ../..

echo '* 2 out of 4 -> commons-codec'
mkdir /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec


echo '* Compiling commons-codec'
cd /home/dorma10/lightweight-effectiveness/projects/commons-codec

echo '* Caching original pom'
cp pom.xml cached_pom.xml

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.Charsets org.apache.commons.codec.CharsetsTest DEFAULT
echo '* Mutating org.apache.commons.codec.Charsets with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.CharsetsTest.txt
mv target/pit-reports target/org.apache.commons.codec.CharsetsTest
cp -r target/org.apache.commons.codec.CharsetsTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.CharsetsTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.StringEncoderComparator org.apache.commons.codec.StringEncoderComparatorTest DEFAULT
echo '* Mutating org.apache.commons.codec.StringEncoderComparator with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.StringEncoderComparatorTest.txt
mv target/pit-reports target/org.apache.commons.codec.StringEncoderComparatorTest
cp -r target/org.apache.commons.codec.StringEncoderComparatorTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.StringEncoderComparatorTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.DecoderException org.apache.commons.codec.DecoderExceptionTest DEFAULT
echo '* Mutating org.apache.commons.codec.DecoderException with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.DecoderExceptionTest.txt
mv target/pit-reports target/org.apache.commons.codec.DecoderExceptionTest
cp -r target/org.apache.commons.codec.DecoderExceptionTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.DecoderExceptionTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.CharEncoding org.apache.commons.codec.CharEncodingTest DEFAULT
echo '* Mutating org.apache.commons.codec.CharEncoding with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.CharEncodingTest.txt
mv target/pit-reports target/org.apache.commons.codec.CharEncodingTest
cp -r target/org.apache.commons.codec.CharEncodingTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.CharEncodingTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.EncoderException org.apache.commons.codec.EncoderExceptionTest DEFAULT
echo '* Mutating org.apache.commons.codec.EncoderException with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.EncoderExceptionTest.txt
mv target/pit-reports target/org.apache.commons.codec.EncoderExceptionTest
cp -r target/org.apache.commons.codec.EncoderExceptionTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.EncoderExceptionTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.net.QCodec org.apache.commons.codec.net.QCodecTest DEFAULT
echo '* Mutating org.apache.commons.codec.net.QCodec with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.net.QCodecTest.txt
mv target/pit-reports target/org.apache.commons.codec.net.QCodecTest
cp -r target/org.apache.commons.codec.net.QCodecTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.net.QCodecTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.net.URLCodec org.apache.commons.codec.net.URLCodecTest DEFAULT
echo '* Mutating org.apache.commons.codec.net.URLCodec with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.net.URLCodecTest.txt
mv target/pit-reports target/org.apache.commons.codec.net.URLCodecTest
cp -r target/org.apache.commons.codec.net.URLCodecTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.net.URLCodecTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.net.Utils org.apache.commons.codec.net.UtilsTest DEFAULT
echo '* Mutating org.apache.commons.codec.net.Utils with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.net.UtilsTest.txt
mv target/pit-reports target/org.apache.commons.codec.net.UtilsTest
cp -r target/org.apache.commons.codec.net.UtilsTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.net.UtilsTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.net.RFC1522Codec org.apache.commons.codec.net.RFC1522CodecTest DEFAULT
echo '* Mutating org.apache.commons.codec.net.RFC1522Codec with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.net.RFC1522CodecTest.txt
mv target/pit-reports target/org.apache.commons.codec.net.RFC1522CodecTest
cp -r target/org.apache.commons.codec.net.RFC1522CodecTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.net.RFC1522CodecTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.net.QuotedPrintableCodec org.apache.commons.codec.net.QuotedPrintableCodecTest DEFAULT
echo '* Mutating org.apache.commons.codec.net.QuotedPrintableCodec with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.net.QuotedPrintableCodecTest.txt
mv target/pit-reports target/org.apache.commons.codec.net.QuotedPrintableCodecTest
cp -r target/org.apache.commons.codec.net.QuotedPrintableCodecTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.net.QuotedPrintableCodecTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.net.PercentCodec org.apache.commons.codec.net.PercentCodecTest DEFAULT
echo '* Mutating org.apache.commons.codec.net.PercentCodec with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.net.PercentCodecTest.txt
mv target/pit-reports target/org.apache.commons.codec.net.PercentCodecTest
cp -r target/org.apache.commons.codec.net.PercentCodecTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.net.PercentCodecTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.net.BCodec org.apache.commons.codec.net.BCodecTest DEFAULT
echo '* Mutating org.apache.commons.codec.net.BCodec with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.net.BCodecTest.txt
mv target/pit-reports target/org.apache.commons.codec.net.BCodecTest
cp -r target/org.apache.commons.codec.net.BCodecTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.net.BCodecTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.DaitchMokotoffSoundex org.apache.commons.codec.language.DaitchMokotoffSoundexTest DEFAULT
echo '* Mutating org.apache.commons.codec.language.DaitchMokotoffSoundex with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.DaitchMokotoffSoundexTest.txt
mv target/pit-reports target/org.apache.commons.codec.language.DaitchMokotoffSoundexTest
cp -r target/org.apache.commons.codec.language.DaitchMokotoffSoundexTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.DaitchMokotoffSoundexTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.Caverphone2 org.apache.commons.codec.language.Caverphone2Test DEFAULT
echo '* Mutating org.apache.commons.codec.language.Caverphone2 with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.Caverphone2Test.txt
mv target/pit-reports target/org.apache.commons.codec.language.Caverphone2Test
cp -r target/org.apache.commons.codec.language.Caverphone2Test /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.Caverphone2Test

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.Metaphone org.apache.commons.codec.language.MetaphoneTest DEFAULT
echo '* Mutating org.apache.commons.codec.language.Metaphone with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.MetaphoneTest.txt
mv target/pit-reports target/org.apache.commons.codec.language.MetaphoneTest
cp -r target/org.apache.commons.codec.language.MetaphoneTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.MetaphoneTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.Nysiis org.apache.commons.codec.language.NysiisTest DEFAULT
echo '* Mutating org.apache.commons.codec.language.Nysiis with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.NysiisTest.txt
mv target/pit-reports target/org.apache.commons.codec.language.NysiisTest
cp -r target/org.apache.commons.codec.language.NysiisTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.NysiisTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.DoubleMetaphone org.apache.commons.codec.language.DoubleMetaphoneTest DEFAULT
echo '* Mutating org.apache.commons.codec.language.DoubleMetaphone with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.DoubleMetaphoneTest.txt
mv target/pit-reports target/org.apache.commons.codec.language.DoubleMetaphoneTest
cp -r target/org.apache.commons.codec.language.DoubleMetaphoneTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.DoubleMetaphoneTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.MatchRatingApproachEncoder org.apache.commons.codec.language.MatchRatingApproachEncoderTest DEFAULT
echo '* Mutating org.apache.commons.codec.language.MatchRatingApproachEncoder with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.MatchRatingApproachEncoderTest.txt
mv target/pit-reports target/org.apache.commons.codec.language.MatchRatingApproachEncoderTest
cp -r target/org.apache.commons.codec.language.MatchRatingApproachEncoderTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.MatchRatingApproachEncoderTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.ColognePhonetic org.apache.commons.codec.language.ColognePhoneticTest DEFAULT
echo '* Mutating org.apache.commons.codec.language.ColognePhonetic with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.ColognePhoneticTest.txt
mv target/pit-reports target/org.apache.commons.codec.language.ColognePhoneticTest
cp -r target/org.apache.commons.codec.language.ColognePhoneticTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.ColognePhoneticTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.RefinedSoundex org.apache.commons.codec.language.RefinedSoundexTest DEFAULT
echo '* Mutating org.apache.commons.codec.language.RefinedSoundex with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.RefinedSoundexTest.txt
mv target/pit-reports target/org.apache.commons.codec.language.RefinedSoundexTest
cp -r target/org.apache.commons.codec.language.RefinedSoundexTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.RefinedSoundexTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.Caverphone1 org.apache.commons.codec.language.Caverphone1Test DEFAULT
echo '* Mutating org.apache.commons.codec.language.Caverphone1 with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.Caverphone1Test.txt
mv target/pit-reports target/org.apache.commons.codec.language.Caverphone1Test
cp -r target/org.apache.commons.codec.language.Caverphone1Test /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.Caverphone1Test

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.Soundex org.apache.commons.codec.language.SoundexTest DEFAULT
echo '* Mutating org.apache.commons.codec.language.Soundex with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.SoundexTest.txt
mv target/pit-reports target/org.apache.commons.codec.language.SoundexTest
cp -r target/org.apache.commons.codec.language.SoundexTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.SoundexTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.bm.BeiderMorseEncoder org.apache.commons.codec.language.bm.BeiderMorseEncoderTest DEFAULT
echo '* Mutating org.apache.commons.codec.language.bm.BeiderMorseEncoder with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.bm.BeiderMorseEncoderTest.txt
mv target/pit-reports target/org.apache.commons.codec.language.bm.BeiderMorseEncoderTest
cp -r target/org.apache.commons.codec.language.bm.BeiderMorseEncoderTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.bm.BeiderMorseEncoderTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.bm.PhoneticEngine org.apache.commons.codec.language.bm.PhoneticEngineTest DEFAULT
echo '* Mutating org.apache.commons.codec.language.bm.PhoneticEngine with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.bm.PhoneticEngineTest.txt
mv target/pit-reports target/org.apache.commons.codec.language.bm.PhoneticEngineTest
cp -r target/org.apache.commons.codec.language.bm.PhoneticEngineTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.bm.PhoneticEngineTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.language.bm.Rule org.apache.commons.codec.language.bm.RuleTest DEFAULT
echo '* Mutating org.apache.commons.codec.language.bm.Rule with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.language.bm.RuleTest.txt
mv target/pit-reports target/org.apache.commons.codec.language.bm.RuleTest
cp -r target/org.apache.commons.codec.language.bm.RuleTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.language.bm.RuleTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.binary.Base32InputStream org.apache.commons.codec.binary.Base32InputStreamTest DEFAULT
echo '* Mutating org.apache.commons.codec.binary.Base32InputStream with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.binary.Base32InputStreamTest.txt
mv target/pit-reports target/org.apache.commons.codec.binary.Base32InputStreamTest
cp -r target/org.apache.commons.codec.binary.Base32InputStreamTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.binary.Base32InputStreamTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.binary.BinaryCodec org.apache.commons.codec.binary.BinaryCodecTest DEFAULT
echo '* Mutating org.apache.commons.codec.binary.BinaryCodec with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.binary.BinaryCodecTest.txt
mv target/pit-reports target/org.apache.commons.codec.binary.BinaryCodecTest
cp -r target/org.apache.commons.codec.binary.BinaryCodecTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.binary.BinaryCodecTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.binary.Base32 org.apache.commons.codec.binary.Base32Test DEFAULT
echo '* Mutating org.apache.commons.codec.binary.Base32 with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.binary.Base32Test.txt
mv target/pit-reports target/org.apache.commons.codec.binary.Base32Test
cp -r target/org.apache.commons.codec.binary.Base32Test /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.binary.Base32Test

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.binary.Base64OutputStream org.apache.commons.codec.binary.Base64OutputStreamTest DEFAULT
echo '* Mutating org.apache.commons.codec.binary.Base64OutputStream with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.binary.Base64OutputStreamTest.txt
mv target/pit-reports target/org.apache.commons.codec.binary.Base64OutputStreamTest
cp -r target/org.apache.commons.codec.binary.Base64OutputStreamTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.binary.Base64OutputStreamTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.binary.Base64InputStream org.apache.commons.codec.binary.Base64InputStreamTest DEFAULT
echo '* Mutating org.apache.commons.codec.binary.Base64InputStream with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.binary.Base64InputStreamTest.txt
mv target/pit-reports target/org.apache.commons.codec.binary.Base64InputStreamTest
cp -r target/org.apache.commons.codec.binary.Base64InputStreamTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.binary.Base64InputStreamTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.binary.BaseNCodec org.apache.commons.codec.binary.BaseNCodecTest DEFAULT
echo '* Mutating org.apache.commons.codec.binary.BaseNCodec with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.binary.BaseNCodecTest.txt
mv target/pit-reports target/org.apache.commons.codec.binary.BaseNCodecTest
cp -r target/org.apache.commons.codec.binary.BaseNCodecTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.binary.BaseNCodecTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.binary.StringUtils org.apache.commons.codec.binary.StringUtilsTest DEFAULT
echo '* Mutating org.apache.commons.codec.binary.StringUtils with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.binary.StringUtilsTest.txt
mv target/pit-reports target/org.apache.commons.codec.binary.StringUtilsTest
cp -r target/org.apache.commons.codec.binary.StringUtilsTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.binary.StringUtilsTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.binary.Hex org.apache.commons.codec.binary.HexTest DEFAULT
echo '* Mutating org.apache.commons.codec.binary.Hex with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.binary.HexTest.txt
mv target/pit-reports target/org.apache.commons.codec.binary.HexTest
cp -r target/org.apache.commons.codec.binary.HexTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.binary.HexTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.binary.Base32OutputStream org.apache.commons.codec.binary.Base32OutputStreamTest DEFAULT
echo '* Mutating org.apache.commons.codec.binary.Base32OutputStream with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.binary.Base32OutputStreamTest.txt
mv target/pit-reports target/org.apache.commons.codec.binary.Base32OutputStreamTest
cp -r target/org.apache.commons.codec.binary.Base32OutputStreamTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.binary.Base32OutputStreamTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.binary.Base64 org.apache.commons.codec.binary.Base64Test DEFAULT
echo '* Mutating org.apache.commons.codec.binary.Base64 with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.binary.Base64Test.txt
mv target/pit-reports target/org.apache.commons.codec.binary.Base64Test
cp -r target/org.apache.commons.codec.binary.Base64Test /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.binary.Base64Test

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.DigestUtils org.apache.commons.codec.digest.DigestUtilsTest DEFAULT
echo '* Mutating org.apache.commons.codec.digest.DigestUtils with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.DigestUtilsTest.txt
mv target/pit-reports target/org.apache.commons.codec.digest.DigestUtilsTest
cp -r target/org.apache.commons.codec.digest.DigestUtilsTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.DigestUtilsTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.PureJavaCrc32C org.apache.commons.codec.digest.PureJavaCrc32CTest DEFAULT
echo '* Mutating org.apache.commons.codec.digest.PureJavaCrc32C with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.PureJavaCrc32CTest.txt
mv target/pit-reports target/org.apache.commons.codec.digest.PureJavaCrc32CTest
cp -r target/org.apache.commons.codec.digest.PureJavaCrc32CTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.PureJavaCrc32CTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.Crypt org.apache.commons.codec.digest.CryptTest DEFAULT
echo '* Mutating org.apache.commons.codec.digest.Crypt with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.CryptTest.txt
mv target/pit-reports target/org.apache.commons.codec.digest.CryptTest
cp -r target/org.apache.commons.codec.digest.CryptTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.CryptTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.B64 org.apache.commons.codec.digest.B64Test DEFAULT
echo '* Mutating org.apache.commons.codec.digest.B64 with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.B64Test.txt
mv target/pit-reports target/org.apache.commons.codec.digest.B64Test
cp -r target/org.apache.commons.codec.digest.B64Test /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.B64Test

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.MurmurHash2 org.apache.commons.codec.digest.MurmurHash2Test DEFAULT
echo '* Mutating org.apache.commons.codec.digest.MurmurHash2 with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.MurmurHash2Test.txt
mv target/pit-reports target/org.apache.commons.codec.digest.MurmurHash2Test
cp -r target/org.apache.commons.codec.digest.MurmurHash2Test /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.MurmurHash2Test

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.HmacUtils org.apache.commons.codec.digest.HmacUtilsTest DEFAULT
echo '* Mutating org.apache.commons.codec.digest.HmacUtils with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.HmacUtilsTest.txt
mv target/pit-reports target/org.apache.commons.codec.digest.HmacUtilsTest
cp -r target/org.apache.commons.codec.digest.HmacUtilsTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.HmacUtilsTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.HmacAlgorithms org.apache.commons.codec.digest.HmacAlgorithmsTest DEFAULT
echo '* Mutating org.apache.commons.codec.digest.HmacAlgorithms with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.HmacAlgorithmsTest.txt
mv target/pit-reports target/org.apache.commons.codec.digest.HmacAlgorithmsTest
cp -r target/org.apache.commons.codec.digest.HmacAlgorithmsTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.HmacAlgorithmsTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.MessageDigestAlgorithms org.apache.commons.codec.digest.MessageDigestAlgorithmsTest DEFAULT
echo '* Mutating org.apache.commons.codec.digest.MessageDigestAlgorithms with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.MessageDigestAlgorithmsTest.txt
mv target/pit-reports target/org.apache.commons.codec.digest.MessageDigestAlgorithmsTest
cp -r target/org.apache.commons.codec.digest.MessageDigestAlgorithmsTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.MessageDigestAlgorithmsTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.UnixCrypt org.apache.commons.codec.digest.UnixCryptTest DEFAULT
echo '* Mutating org.apache.commons.codec.digest.UnixCrypt with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.UnixCryptTest.txt
mv target/pit-reports target/org.apache.commons.codec.digest.UnixCryptTest
cp -r target/org.apache.commons.codec.digest.UnixCryptTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.UnixCryptTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.Md5Crypt org.apache.commons.codec.digest.Md5CryptTest DEFAULT
echo '* Mutating org.apache.commons.codec.digest.Md5Crypt with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.Md5CryptTest.txt
mv target/pit-reports target/org.apache.commons.codec.digest.Md5CryptTest
cp -r target/org.apache.commons.codec.digest.Md5CryptTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.Md5CryptTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.Sha2Crypt org.apache.commons.codec.digest.Sha2CryptTest DEFAULT
echo '* Mutating org.apache.commons.codec.digest.Sha2Crypt with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.Sha2CryptTest.txt
mv target/pit-reports target/org.apache.commons.codec.digest.Sha2CryptTest
cp -r target/org.apache.commons.codec.digest.Sha2CryptTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.Sha2CryptTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.MurmurHash3 org.apache.commons.codec.digest.MurmurHash3Test DEFAULT
echo '* Mutating org.apache.commons.codec.digest.MurmurHash3 with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.MurmurHash3Test.txt
mv target/pit-reports target/org.apache.commons.codec.digest.MurmurHash3Test
cp -r target/org.apache.commons.codec.digest.MurmurHash3Test /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.MurmurHash3Test

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.XXHash32 org.apache.commons.codec.digest.XXHash32Test DEFAULT
echo '* Mutating org.apache.commons.codec.digest.XXHash32 with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.XXHash32Test.txt
mv target/pit-reports target/org.apache.commons.codec.digest.XXHash32Test
cp -r target/org.apache.commons.codec.digest.XXHash32Test /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.XXHash32Test

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-codec org.apache.commons.codec.digest.PureJavaCrc32 org.apache.commons.codec.digest.PureJavaCrc32Test DEFAULT
echo '* Mutating org.apache.commons.codec.digest.PureJavaCrc32 with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.codec.digest.PureJavaCrc32Test.txt
mv target/pit-reports target/org.apache.commons.codec.digest.PureJavaCrc32Test
cp -r target/org.apache.commons.codec.digest.PureJavaCrc32Test /home/dorma10/lightweight-effectiveness/mutation_results/commons-codec

rm -rf target/org.apache.commons.codec.digest.PureJavaCrc32Test
echo '* Restoring original pom'
rm -rf pom.xml
mv cached_pom.xml pom.xml
cd ../..

echo '* 3 out of 4 -> commons-exec'
mkdir /home/dorma10/lightweight-effectiveness/mutation_results/commons-exec


echo '* Compiling commons-exec'
cd /home/dorma10/lightweight-effectiveness/projects/commons-exec

echo '* Caching original pom'
cp pom.xml cached_pom.xml

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-exec org.apache.commons.exec.LogOutputStream org.apache.commons.exec.LogOutputStreamTest DEFAULT
echo '* Mutating org.apache.commons.exec.LogOutputStream with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.exec.LogOutputStreamTest.txt
mv target/pit-reports target/org.apache.commons.exec.LogOutputStreamTest
cp -r target/org.apache.commons.exec.LogOutputStreamTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-exec

rm -rf target/org.apache.commons.exec.LogOutputStreamTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-exec org.apache.commons.exec.DefaultExecutor org.apache.commons.exec.DefaultExecutorTest DEFAULT
echo '* Mutating org.apache.commons.exec.DefaultExecutor with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.exec.DefaultExecutorTest.txt
mv target/pit-reports target/org.apache.commons.exec.DefaultExecutorTest
cp -r target/org.apache.commons.exec.DefaultExecutorTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-exec

rm -rf target/org.apache.commons.exec.DefaultExecutorTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-exec org.apache.commons.exec.CommandLine org.apache.commons.exec.CommandLineTest DEFAULT
echo '* Mutating org.apache.commons.exec.CommandLine with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.exec.CommandLineTest.txt
mv target/pit-reports target/org.apache.commons.exec.CommandLineTest
cp -r target/org.apache.commons.exec.CommandLineTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-exec

rm -rf target/org.apache.commons.exec.CommandLineTest

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py commons-exec org.apache.commons.exec.environment.EnvironmentUtils org.apache.commons.exec.environment.EnvironmentUtilsTest DEFAULT
echo '* Mutating org.apache.commons.exec.environment.EnvironmentUtils with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.apache.commons.exec.environment.EnvironmentUtilsTest.txt
mv target/pit-reports target/org.apache.commons.exec.environment.EnvironmentUtilsTest
cp -r target/org.apache.commons.exec.environment.EnvironmentUtilsTest /home/dorma10/lightweight-effectiveness/mutation_results/commons-exec

rm -rf target/org.apache.commons.exec.environment.EnvironmentUtilsTest
echo '* Restoring original pom'
rm -rf pom.xml
mv cached_pom.xml pom.xml
cd ../..

echo '* 4 out of 4 -> cors-filter'
mkdir /home/dorma10/lightweight-effectiveness/mutation_results/cors-filter


echo '* Compiling cors-filter'
cd /home/dorma10/lightweight-effectiveness/projects/cors-filter

echo '* Caching original pom'
cp pom.xml cached_pom.xml

python3 /home/dorma10/lightweight-effectiveness/effectiveness/mutation/pom_changer.py cors-filter org.ebaysf.web.cors.CORSFilter org.ebaysf.web.cors.CORSFilterTest DEFAULT
echo '* Mutating org.ebaysf.web.cors.CORSFilter with operator DEFAULT'
timeout 20m mvn org.pitest:pitest-maven:mutationCoverage -X -DoutputFormats=HTML --log-file ../../logs/org.ebaysf.web.cors.CORSFilterTest.txt
mv target/pit-reports target/org.ebaysf.web.cors.CORSFilterTest
cp -r target/org.ebaysf.web.cors.CORSFilterTest /home/dorma10/lightweight-effectiveness/mutation_results/cors-filter

rm -rf target/org.ebaysf.web.cors.CORSFilterTest
echo '* Restoring original pom'
rm -rf pom.xml
mv cached_pom.xml pom.xml
cd ../..

mv /home/dorma10/lightweight-effectiveness/mutation_results /home/dorma10/lightweight-effectiveness/mutation_results-DEFAULT
