.PHONY: all
all: src/schema_generated.rs tools/schema_generated.py

.PHONY: clean
clean:
	rm src/*_generated.rs tools/*_generated.py

.PHONY: check
check:
	cargo fmt --check
	cargo test

src/schema_generated.rs: src/schema.fbs
	flatc -o src/ --rust src/schema.fbs
	cargo fmt
	(echo "#![allow(clippy::all)]" && cat src/schema_generated.rs) > src/schema_generated.rs.tmp
	mv src/schema_generated.rs.tmp src/schema_generated.rs

tools/schema_generated.py: src/schema.fbs
	flatc -o tools/ --gen-onefile --python src/schema.fbs
