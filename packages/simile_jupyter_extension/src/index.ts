import type {
	JupyterFrontEnd,
	JupyterFrontEndPlugin,
} from "@jupyterlab/application";

/**
 * Initialization data for the simile_lexer extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
	id: "simile_lexer:plugin",
	description: "CodeMirror lexer for Simile",
	autoStart: true,
	activate: (app: JupyterFrontEnd) => {
		console.log("JupyterLab extension simile_lexer is activated!");
	},
};

export default plugin;
