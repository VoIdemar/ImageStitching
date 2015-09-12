@echo off

echo ===============================================================================
echo.
echo   Building dot-files for Python profiling statistics...
echo.
echo ===============================================================================

for %%f in (*.raw_stats) do (
	echo    --- Profiling %%f...
	gprof2dot %%f -f pstats -c custom > %%f.dot
	echo        dot-graph: %%f.dot
	dot -Tpng %%f.dot -o %%f.png
	echo        dot-graph as image: %%f.png
	echo        Finished!
)

dot -Tpng ololo.dot -o ololo.png

echo ===============================================================================
echo.
echo   Done!
echo.
echo ===============================================================================

pause