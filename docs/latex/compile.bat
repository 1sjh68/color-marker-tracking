@echo off
chcp 65001 >nul
echo ========================================
echo  LaTeX 编译脚本 (XeLaTeX)
echo ========================================

cd /d "%~dp0"

echo.
echo [1/3] 第一次编译...
xelatex -interaction=nonstopmode optics_term_paper.tex

echo.
echo [2/3] 第二次编译 (生成目录)...
xelatex -interaction=nonstopmode optics_term_paper.tex

echo.
echo [3/3] 清理临时文件...
del /q *.aux *.log *.toc *.out 2>nul

echo.
echo ========================================
echo  编译完成！
echo  输出文件: optics_term_paper.pdf
echo ========================================
pause
