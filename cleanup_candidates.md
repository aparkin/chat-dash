# Cleanup Candidates

This document lists files that are candidates for removal or cleanup, organized by priority and risk level.

## High Priority - Safe to Remove
These files are definitely obsolete or have been superseded by new implementations:

- `tests/test_wqp.py`
  - Reason: Uses old direct API testing approach
  - Replaced by: Our new direct API implementation in `test_direct_api.py`

- `tests/test_hyriver.py`
  - Reason: Uses deprecated HyRiver library
  - Replaced by: Direct API implementation in `services/usgs/direct_api.py`

- `services/usgs/cache/`
  - Files:
    - `sites.json`
    - `parameters.json`
  - Reason: Caching has been removed in favor of direct API calls
  - Replaced by: Real-time API calls

- `test_query.py` (in root directory)
  - Reason: Standalone test file in root directory
  - Should be: In tests directory if still needed

## Medium Priority - Review Before Removal
These files may be obsolete but should be reviewed before removal:

- `tests/usgs/test_service.py`
  - Reason: Potentially redundant with new interactive test
  - Review: Check if any unique test cases should be migrated

- `tests/usgs/test_water_quality.py`
  - Reason: May be consolidated with other tests
  - Review: Compare with `test_direct_api.py` and interactive test

- `tests/test_site_data.py`
  - Reason: May be redundant with new implementation
  - Review: Check for any unique functionality

- `services/uniprot/test_script.py`
  - Reason: Appears to be development/debugging script
  - Review: Verify with UniProt service maintainer

- `services/uniprot/test_data_manager.py`
  - Reason: May be development-only code
  - Review: Check if used by UniProt service

## Low Priority - Investigate
These files need investigation before making a decision:

- `tests/test_weaviate.py`
  - Action: Check if Weaviate is still used anywhere in the codebase
  - Note: May be part of search functionality

- `cache/aiohttp_cache.sqlite`
  - Action: Verify if any services still use aiohttp caching
  - Note: May be used by other services

## Keep for Reference
These files should be kept as they're part of the current implementation:

- `tests/test_direct_api.py`
- `tests/test_usgs_water_interactive.py`
- `services/usgs/direct_api.py`
- `services/usgs/water_quality.py`
- `services/usgs/service.py`

## Next Steps

1. Review this list and mark files that are approved for removal
2. Back up files before deletion
3. Remove files in batches, starting with High Priority
4. Test system after each batch removal
5. Update documentation to reflect changes

## Notes

- Some test files might contain useful test cases that should be migrated to the new test suite
- Consider creating a temporary backup directory for removed files
- Update any documentation that might reference removed files 