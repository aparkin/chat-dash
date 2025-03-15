# Comprehensive Cleanup List

This document lists files and directories that are candidates for cleanup, organized by priority and risk level.

## High Priority - Safe to Remove or Archive

### Temporary Files
- `tmp.txt` (368KB) - Temporary text file
- `tmp.json` (379KB) - Temporary JSON file
- `tmp.zip` (66KB) - Temporary ZIP file
- `.coverage` and `.coverage.APArkin-M72.36748.XNfcWorx` - Coverage report files that should not be in version control

### Test Data Files
- `narrowresult.csv` (839KB) - Appears to be test/temporary data
- `test_streamflow.csv` (1.3KB) - Test data
- `test_params.csv` (1.4MB) - Test parameters file
- `water_quality_data_point.tsv` (6.0KB) - Sample data point
- `water_quality_metadata_point.json` (4.5KB) - Sample metadata point
- `water_quality_metadata.json` (16KB) - Sample metadata
- `water_quality_data.tsv` (1.9MB) - Large test data file

### Obsolete Files (from cleanup_candidates.md)
- `tests/test_wqp.py` - Uses old direct API testing approach
- `tests/test_hyriver.py` - Uses deprecated HyRiver library
- `services/usgs/cache/sites.json` - Caching has been removed in favor of direct API calls
- `services/usgs/cache/parameters.json` - Caching has been removed in favor of direct API calls
- `test_query.py` (in root directory) - Standalone test file in root directory

### Versioned Files
The `versions/` directory contains 28 historical versions of ChatDash.py. These should be:
- Archived (preferably in a proper version control system like Git)
- Removed from the main workspace
- Keep only the most recent version if needed for quick reference

### Old Manager
The `old_analysis_manager/` directory appears to be an obsolete version:
- `analysis_service.py`
- `shared_instances.py`
- `analysis_callbacks.py`
- `analysis_ui.py`
- `analysis_manager.py`

## Medium Priority - Review Before Removal

### Potentially Redundant Files
- `weaviate_debug.json` (566KB) - Debug file for Weaviate
- `debug_weaviate.py` (1.1KB) - Debug script for Weaviate
- `run_usgs_service.py` (2.5KB) - May be a standalone script that could be moved to tests or utilities
- `schema_old.md` (1.7KB) - Old schema file

### Weaviate Integration
There appear to be two systems for Weaviate integration:
- `WeaviateQueryManager/` - Older version (last modified January 2024)
- `weaviate_manager/` - Newer version (last modified February 2024)

Review whether both are needed or if one can be removed.

### Log Files
- `weaviate_import.log` (17MB) - Large log file that should not be in version control

### Cache Files
- `cache/aiohttp_cache.sqlite` (90MB) - Large cache file that should be in `.gitignore`

## Low Priority - Investigate

- `database_manager.py` vs services integration - Check if this standalone manager is still used
- `.DS_Store` - macOS system file that should be in `.gitignore`
- `schema.md` vs current schema implementation - Verify if this is current

## Action Plan for File Cleanup

1. **Create Archive Directory**:
   ```
   mkdir -p archive/tmp_files
   mkdir -p archive/test_data
   mkdir -p archive/old_systems
   mkdir -p archive/logs
   ```

2. **Move Temporary Files**:
   ```
   mv tmp.txt archive/tmp_files/
   mv tmp.json archive/tmp_files/
   mv tmp.zip archive/tmp_files/
   ```

3. **Move Test Data**:
   ```
   mv narrowresult.csv archive/test_data/
   mv test_streamflow.csv archive/test_data/
   mv test_params.csv archive/test_data/
   mv water_quality_*.* archive/test_data/
   ```

4. **Archive Old Systems**:
   ```
   mv old_analysis_manager/ archive/old_systems/
   ```

5. **Review Weaviate Integration**:
   Compare `WeaviateQueryManager/` and `weaviate_manager/` to determine which to keep.

6. **Update .gitignore**:
   Ensure `.gitignore` includes:
   ```
   .DS_Store
   *.sqlite
   .coverage*
   *.log
   ```

7. **Handle Versions**:
   Discuss whether to keep the `versions/` directory or use Git for version control properly.

## Notes

- Ensure any files scheduled for removal are not referenced elsewhere in the codebase
- Consider creating a backup before removing any files
- Update documentation to reflect changes
- After cleanup, run tests to ensure functionality is preserved 