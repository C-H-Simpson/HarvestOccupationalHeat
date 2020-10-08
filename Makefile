SETTINGS=mysettings/CMIP6_choices.py mysettings/geography.py mysettings/storage_settings.py


# This is a little hard, because we need to iterate over a list
# Put the list in a file and iterate in there
# Then put the list file as a dependency
# Rather than tracking dependencies for each run
logs/WBGT_CMIP6.log: $(SETTINGS) src/WBGT/Calculate_WBGT.py src/WBGT/Calculate_WBGT_CMIP6.py 
	python -u src/WBGT/Calculate_WBGT_CMIP6.py

